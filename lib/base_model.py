import os,sys,shutil,mmap
import cPickle as pickle

from time import time
import numpy as np

import tensorflow as tf

import config

from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score

sz_float = 4

def get_training_sample(d=16384, normalize=True):
    cachex = config.tmp + 'Xtrain_ext{}.npy'.format(d)
    cachey = config.tmp + 'Ytrain_ext{}.npy'.format(d)
    if os.path.exists(cachex) and os.path.exists(cachey):
        Xtrain = np.load(cachex)
        Ytrain = np.load(cachey)
    else:
        raise Exception('Unimplemented')

    if normalize:
        Xtrain = Xtrain.reshape(len(Xtrain),1,d,1)
        for i in range(len(Xtrain)):
            Xtrain[i] /= np.linalg.norm(Xtrain[i]) + config.epsilon

    return Xtrain,Ytrain

class Model(object):
    def __init__(self, labels, checkpoint_path, init=False, window=16384, outputs=1, stride=512, normalize=True, gpu_memory_growth=False, extended_test_set=True, use_mirex=False, pitch_transforms=0, mmap=True, batch_size=150, jitter=0, breakdowns=True, restrict=True):
        self.labels = labels
        self.batch_size = batch_size

        self.window = window
        self.cp = checkpoint_path
        self.normalize = normalize
        self.gpu_memory_growth = gpu_memory_growth
        self.pitch_transforms = pitch_transforms
        self.jitter = jitter
        self.mmap = mmap
        self.breakdowns = breakdowns

        # use the small or big test set
        if extended_test_set:
            self.test_ids = config.test_ids_ext
        else:
            self.test_ids = config.test_ids

        # include the mirex dev set in test results
        if use_mirex:
            self.test_ids = self.test_ids + config.mirex_id

        self.train_ids = [rec_id for rec_id in self.labels.keys() if rec_id not in self.test_ids]

        if restrict:
            # possible notes are range [self.base_note,self.base_note+self.m)
            self.base_note = 128
            max_note = 0
            for rec_id in self.train_ids:
                for label in sorted(labels[rec_id]):
                    note = label.data[1]
                    if note < self.base_note:
                        self.base_note = note
                    if note > max_note:
                        max_note = note
            self.base_note -= pitch_transforms
            max_note += self.pitch_transforms
            self.m = max_note-self.base_note
        else:
            self.base_note = 0
            self.m = 128

        # for multiple output predictions
        self.stride = stride
        self.out = outputs

        self.weights = dict()

        self.iter = 0

        self.stats = dict()
        # (print?,format string, values)
        self.stats['iter'] = [False,'{:<8}',[]]
        self.stats['time'] = [True,'{:<8.0f}',[]]
        self.stats['utime'] = [True,'{:<8.0f}',[]]
        self.stats['lr'] = [False,'{:<8.6f}',[]]
        self.stats['mse_train'] = [True,'{:<16.6f}',[]]
        self.stats['mse_test'] = [True,'{:<16.6f}',[]]
        self.stats['avp_train'] = [True,'{:<16.6f}',[]]
        self.stats['avp_test'] = [True,'{:<16.6f}',[]]
        #self.stats['mse_t-1'] = [True,'{:<8.4f}',[]]
        #self.stats['avp_t-1'] = [True,'{:<8.4f}',[]]
        #self.stats['mse_t-5'] = [True,'{:<8.4f}',[]]
        #self.stats['avp_t-5'] = [True,'{:<8.4f}',[]]

        if self.breakdowns:
            for rec_id in self.test_ids:
                self.stats['mse_test_' + str(rec_id)] = [False,'{:<8.4}',[]]
                self.stats['avp_test_' + str(rec_id)] = [False,'{:<8.4}',[]]

        # see shallow_models/mlp_small_diagnostics for how to do this
        #self.stats['|Eg_test|^2'] = [False,'{:<16.6f}',[]]
        #self.stats['|Eg_train|^2'] = [False,'{:<16.6f}',[]]
        #self.stats['E|g_test|^2'] = [False,'{:<16.6f}',[]]
        #self.stats['E|g_train|^2'] = [False,'{:<16.6f}',[]]

        self.coord = None

        self.init = init
        self.define_graph()
        self.start()

    def register_weights(self, w, name, average=0.):
        self.stats['n'+name] = [False,'{:<8.3f}',[]] 
        self.weights[name] = w

        if average>0: 
            wavg = tf.Variable(w.initialized_value())
            #wavg = tf.Variable(tf.zeros(w.get_shape()))
            self.stats['navg_'+name] = [False,'{:<8.3f}',[]] 
            self.weights['avg_'+name] = wavg
            self.averages.append(tf.assign(wavg, average*wavg + (1-average)*w))
            return wavg
        else:
            return None

    def define_graph(self):
        tf.reset_default_graph()
        tf.set_random_seed(999)

        self.averages = []

        with tf.variable_scope('data_queue'):
            self.xb = tf.placeholder(tf.float32, shape=[None,1,self.window+(self.out-1)*self.stride,1])
            self.yb = tf.placeholder(tf.float32, shape=[None, self.out,self.m])
    
            self.queue = tf.FIFOQueue(capacity=20*self.batch_size,
                                 dtypes=[tf.float32,tf.float32],
                                 shapes=[[1,self.window+(self.out-1)*self.stride,1],[self.out,self.m]])
            self.enqueue_op = self.queue.enqueue_many([self.xb, self.yb])
            self.xq, self.yq = self.queue.dequeue_many(self.batch_size)

        with tf.variable_scope('direct_data'):
            self.xd = tf.placeholder(tf.float32, shape=[None,1,self.window,1])
            self.yd = tf.placeholder(tf.float32, shape=[None, self.m])

        # subclasses must define these quantities
        self.y_direct = None       # model predictions with direct evaluation
        self.loss_direct = None    # model loss with direct evaluation
        self.loss = None           # model loss

    def get_data(self,rec_id,s,pitch_shift,scaling_factor):
        if self.mmap:
            x = np.frombuffer(self.data[rec_id][0][s*sz_float:int(s+scaling_factor*self.window)*sz_float], dtype=np.float32).copy()
        else:
            fid,length = self.data[rec_id]
            f = open(fid, 'rb')
            f.seek(s*sz_float, os.SEEK_SET)
            #x = np.frombuffer(f.read(int(scaling_factor*self.window)*sz_float), dtype=np.float32).copy()
            x = np.fromfile(f, dtype=np.float32, count=int(scaling_factor*self.window))
            f.close()

        if self.normalize: x /= np.linalg.norm(x) + config.epsilon

        xp = np.arange(self.window,dtype=np.float32)
        x = np.interp(scaling_factor*xp,np.arange(len(x),dtype=np.float32),x).astype(np.float32)

        y = np.zeros(self.m)
        for label in self.labels[rec_id][s+scaling_factor*self.window/2]:
            y[label.data[1]+pitch_shift-self.base_note] = 1

        return x,y

    def start(self):
        def fetch_data():
            xmb = np.empty([self.batch_size,1,self.window,1],dtype=np.float32)
            ymb = np.empty([self.batch_size,1,self.m],dtype=np.float32)
            while not self.coord.should_stop():
                rec_ids = [self.train_ids[r] for r in np.random.randint(0,len(self.train_ids),self.batch_size)]

                if self.pitch_transforms > 0:
                    transform_type = np.random.randint(-self.pitch_transforms,self.pitch_transforms,self.batch_size)

                if self.jitter > 0:
                    jitter_amount = np.random.uniform(-self.jitter,self.jitter,self.batch_size)

                for i in range(self.batch_size):

                    scaling_factor = 1
                    shift = 0
                    if self.pitch_transforms > 0:
                        shift = transform_type[i]
                        if self.jitter > 0:
                            scaling_factor = (2.**((shift+jitter_amount[i])/12.))
                        else:
                            scaling_factor = (2.**(shift/12.))

                    s = np.random.randint(0,self.data[rec_ids[i]][1]-scaling_factor*self.window)
                    xmb[i,0,:,0],ymb[i,0,:] = self.get_data(rec_ids[i],s,pitch_shift=shift,scaling_factor=scaling_factor)

                self.sess.run(self.enqueue_op, feed_dict={self.xb: xmb, self.yb: ymb})

        init_op = tf.global_variables_initializer()

        tfconfig = tf.ConfigProto()
        tfconfig.gpu_options.allow_growth=self.gpu_memory_growth
        self.sess = tf.Session(config=tfconfig)

        self.sess.run(init_op)
        self.precondition()

        if self.init:
            if os.path.exists(self.cp):
                shutil.rmtree(self.cp)
                os.mkdir(self.cp)
            else:
                os.mkdir(self.cp)
            self.checkpoint()
            self.init = False
        else:
            self.restore_checkpoint()

        # map the dataset
        self.data = dict()
        self.files = []
        for record in os.listdir(config.records_path):
            if self.mmap:
                fd = os.open(config.records_path + record, os.O_RDONLY)
                buff = mmap.mmap(fd, 0, mmap.MAP_SHARED, mmap.PROT_READ)
                self.data[int(record[:-4])] = (buff, len(buff)/sz_float)
                self.files.append(fd)
            else:
                f = open(config.records_path + record)
                self.data[int(record[:-4])] = (config.records_path + record,os.fstat(f,fileno()).st_size/sz_float)
                f.close()
        
        self.coord = tf.train.Coordinator()
        self.qr = tf.train.QueueRunner(self.queue, [tf.py_func(fetch_data,[],[])] * 8)
        self.workers = self.qr.create_threads(self.sess, coord=self.coord, start=True)

    def precondition(self):
        pass

    def stop(self):
        if self.coord:
            self.coord.request_stop()
            self.coord.join(self.workers) 
            self.coord = None

        if self.mmap:
            for mm in self.data.values():
                mm[0].close()
            for fd in self.files:
                os.close(fd)
            self.data = dict()
            self.files = []

    def checkpoint(self):
        for stat,value in self.stats.items():
            with open(self.cp + stat + '.npy', 'wb') as f:
                np.save(f,value[2]) 

        for name, w in self.weights.items():
            with open(self.cp + name + '.npy','wb') as f:
                np.save(f,w.eval(session=self.sess))

    def restore_checkpoint(self):
        for stat in self.stats:
            with open(self.cp + stat + '.npy', 'rb') as f:
                self.stats[stat][2] = list(np.load(f))

        for name, w in self.weights.items():
            with open(self.cp + name + '.npy','rb') as f:
                self.sess.run(w.assign(np.load(f)))

        if len(self.stats['iter'][2]) > 0:
            self.iter = self.stats['iter'][2][-1]

    def sample_records(self, rec_ids, count, fixed_stride=-1, pitch_shift=0):
        mse = 0.
        mse_breakdown = dict()
        avp_breakdown = dict()
        offset = 44100
        Yhat = np.empty((count*len(rec_ids),self.m))
        Yall = np.empty((count*len(rec_ids),self.m))
        sf = 2.**(pitch_shift/12.)
        for i in range(len(rec_ids)):
            X = np.zeros([count,1,self.window,1])
            Y = np.zeros([count,self.m])

            stride = (self.data[rec_ids[i]][1]-offset-int(sf*self.window))/count if fixed_stride==-1 else fixed_stride
            for j in range(count):
                X[j,0,:,0],Y[j] = self.get_data(rec_ids[i],offset+j*stride,pitch_shift=int(round(pitch_shift)),scaling_factor=sf)

            this_mse = self.mse(X,Y)
            mse_breakdown[rec_ids[i]] = this_mse
            mse += this_mse/len(rec_ids)

            this_Yhat = self.predict(X)
            Yall[count*i:count*(i+1)] = Y
            Yhat[count*i:count*(i+1)] = this_Yhat
            avp_breakdown[rec_ids[i]] = average_precision_score(Y.flatten(),this_Yhat.flatten())

        return mse, Yhat, Yall, mse_breakdown, avp_breakdown

    # call last if subclassing to ensure proper ptime/utime updates
    def update_status(self,ptime,utime,lr):
        mse_test, Yhat, Y, mse_breakdown, avp_breakdown = self.sample_records(self.test_ids, 1000)
        avp_test = average_precision_score(Y.flatten(),Yhat.flatten())
        del Yhat, Y

        #mse_testm1, Yhat, Y, _, _ = self.sample_records(self.test_ids, 500, pitch_shift=-1)
        #avp_testm1 = average_precision_score(Y.flatten(),Yhat.flatten())

        #mse_testm5, Yhat, Y, _, _ = self.sample_records(self.test_ids, 500, pitch_shift=-5)
        #avp_testm5 = average_precision_score(Y.flatten(),Yhat.flatten())

        X,Y = get_training_sample(self.window)
        Y = Y[:,self.base_note:self.base_note+self.m]
        mse_train = self.mse(X,Y)
        avp_train = average_precision_score(Y.flatten(),self.predict(X).flatten())
        del X,Y

        weight_norms = dict()
        for name, w in self.weights.items():
            weight_norms[name] = np.mean(np.linalg.norm(w.eval(session=self.sess),axis=0))

        # do all the assignments at the end so hopefully we don't get interrupted mid-update
        for name, norm in weight_norms.items():
            self.stats['n'+name][2].append(norm)
        self.stats['mse_test'][2].append(mse_test)
        self.stats['avp_test'][2].append(avp_test)
        #self.stats['mse_t-1'][2].append(mse_testm1)
        #self.stats['avp_t-1'][2].append(avp_testm1)
        #self.stats['mse_t-5'][2].append(mse_testm5)
        #self.stats['avp_t-5'][2].append(avp_testm5)
        self.stats['mse_train'][2].append(mse_train)
        self.stats['avp_train'][2].append(avp_train)
        self.stats['iter'][2].append(self.iter)
        self.stats['lr'][2].append(lr)
        self.stats['time'][2].append(time() - ptime)
        self.stats['utime'][2].append(time() - utime)

        if self.breakdowns:
            for rec_id in self.test_ids:
                self.stats['mse_test_' + str(rec_id)][2].append(mse_breakdown[rec_id])
                self.stats['avp_test_' + str(rec_id)][2].append(avp_breakdown[rec_id])

    def status_header(self):
        return '\t'.join(sorted([key for key,val in self.stats.items() if val[0]]))

    def status(self):
        status_str = ''
        for key,val in sorted(self.stats.items()):
            if val[0]:
                format_str = val[1]
                current_val = val[2][-1]
                status_str += format_str.format(current_val)
        return status_str

    def mse(self,X,Y):
        se = 0.
        subdiv = 100
        subset = X.shape[0]/subdiv
        for j in range(subdiv):
            se += self.sess.run(self.loss_direct,
                           feed_dict={self.xd: X[subset*j:subset*(j+1)],
                                      self.yd: Y[subset*j:subset*(j+1)]})/subset

        return se/float(subdiv)

    def predict(self,X):
        Yhat = np.empty((len(X),self.m))
        subdiv = 100
        subset = X.shape[0]/subdiv
        for j in range(subdiv):
            Yhat[subset*j:subset*(j+1)] = \
            self.sess.run(self.y_direct, feed_dict={self.xd: X[subset*j:subset*(j+1)]})

        return Yhat

