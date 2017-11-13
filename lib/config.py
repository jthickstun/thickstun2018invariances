# data location
records_path = 'data/records/'
labels_path = 'data/labels_mirex.pckl'
tmp = 'data/'

# number of gpus on this system (for scheduling)
gpu_count = 4

# fudge factor for normalization
epsilon = 10e-8

# mirex dataset
mirex_id = (2718,)
mirex_compid = ('3039',)

# small test set
#   2303 - Bach solo piano
#   2382 - Beethoven string quartet
#   1819 - Mozart wind quintet
test_ids = (2303,2382,1819)
test_compids = ('241','647','2589')

# extended test set
#   2298 - Bach solo cello
#   2191 - Bach solo violin
#   2556 - Beethoven solo piano (prestissimo)
#   2416 - Beethoven wind sextet (doubled clarinet, horn, bassoon)
#   2628 - Beethoven violin sonata (violin + piano)
#   1759 - Schubert solo piano
#   2106 - Haydn string quartet
test_ids_ext = (2303,2382,1819,2298,2191,2556,2416,2628,1759,2106)
test_compids_ext = ('241','647','2589','42','174','476','838','821','2344','1408','3039')
