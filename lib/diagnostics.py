import matplotlib.pyplot as plt
from matplotlib import colors
import numpy as np
import mir_eval
import config
import mido

from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score

PREDICTIONS = 0
RAW_PREDICTIONS = 1
GROUND = 2
BREAKDOWN = 3
MIREX = -1
def plot_transcription(model,record,version,threshold=.5):
    mse_test, Yhat, Y, mse_breakdown, avp_breakdown = model.sample_records([model.test_ids[record]], 4000, fixed_stride=512)
    visual_predict = np.flipud(Yhat.T)>threshold

    fig, ax = plt.subplots(figsize=(150,7))
    if version == PREDICTIONS:
        ax.imshow(visual_predict,interpolation='none',cmap='Greys',aspect=8)
    elif version == RAW_PREDICTIONS:
        visual = np.flipud(Yhat.T)
        ax.imshow(visual,interpolation='none',cmap='Greys',aspect=8)
    elif version == GROUND or version == BREAKDOWN:
        visual_ground = np.flipud(Y.T)

        if version == GROUND:
            ax.imshow(visual_ground,interpolation='none',cmap='Greys',aspect=8)
        else:
            tp = visual_predict*visual_ground
            tn = (1-visual_predict)*(1-visual_ground)
            fp = visual_predict*(1-visual_ground)
            fn = (1-visual_predict)*visual_ground

            cmap = colors.ListedColormap(['white','black','orange','red'])
            bounds = [0,1,2,3,4]
            norm = colors.BoundaryNorm(bounds, cmap.N)
            error_summary = 0*tn + 1*tp + 2*fn + 3*fp
            ax.imshow(error_summary,interpolation='none',cmap=cmap,norm=norm,aspect=8)

def mirex_statistics(model,record,threshold=.5,initial=False):
    if initial:
        mse_test, Yhat, Y, mse_breakdown, avp_breakdown = model.sample_records([model.test_ids[record]], 4000, fixed_stride=512)
    else:
        mse_test, Yhat, Y, mse_breakdown, avp_breakdown = model.sample_records([model.test_ids[record]], 1000)

    Yhatpred = Yhat>threshold

    Yhatlist = []
    Ylist = []
    for i in range(len(Yhatpred)):
        fhat = []
        f = []
        for note in range(model.m):
            if Yhatpred[i][note] == 1:
                fhat.append(440.*2**(((note+model.base_note) - 69.)/12.))
            if Y[i][note] == 1:
                f.append(440.*2**(((note+model.base_note) - 69.)/12.))

        Yhatlist.append(np.array(fhat))
        Ylist.append(np.array(f))

    avp = average_precision_score(Y.flatten(),Yhat.flatten())
    
    P,R,Acc,Esub,Emiss,Efa,Etot,cP,cR,cAcc,cEsub,cEmiss,cEfa,cEtot = \
    mir_eval.multipitch.metrics(np.arange(len(Ylist))/100.,Ylist,np.arange(len(Yhatlist))/100.,Yhatlist)

    print 'AvgP\tP\tR\tAcc\tETot\tESub\tEmiss\tEfa'
    print '{:.2f}\t{:.2f}\t{:.2f}\t{:.2f}\t{:.2f}\t{:.2f}\t{:.2f}\t{:.2f}'.format(100*avp,100*P,100*R,Acc,Etot,Esub,Emiss,Efa)

    return avp,P,R,Acc,Etot

def pr_curve(model,record):
    mse_test, Yhat, Y, mse_breakdown, avp_breakdown = model.sample_records([model.test_ids[record]], 1000)

    P,R,_ = precision_recall_curve(Y.flatten(),Yhat.flatten())
    plt.plot(R,P)

def midi_transcription(model,record,outfile='pred.mid',threshold=.5):
    mse_test, Yhat, Y, mse_breakdown, avp_breakdown = model.sample_records([model.test_ids[record]], 4000, fixed_stride=512)
    Yhat = Yhat>threshold

    mid = mido.MidiFile()
    track = mido.MidiTrack()
    mid.tracks.append(track)

    Ypred = Yhat>.4

    notes = np.zeros(128)
    elapsed_t = 0
    for i in range(len(Ypred)):
        for j in range(128):
            # onset
            if Ypred[i,j] == 1 and notes[j] == 0:
                track.append(mido.Message('note_on', note=j, time=elapsed_t))
                notes[j] = 1
                elapsed_t = 0
            
            # offset
            if Ypred[i,j] == 0 and notes[j] == 1:
                track.append(mido.Message('note_off', note=j, time=elapsed_t))
                notes[j] = 0
                elapsed_t = 0
            
        elapsed_t += 10
    
    for j in range(128):
        if notes[j] == 1:
            track.append(mido.Message('note_off', note=j, time=elapsed_t))
            notes[j] = 0
            elapsed_t = 0
    
    mid.save(outfile) 
