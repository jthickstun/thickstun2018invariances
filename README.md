# thickstun2018invariances

Experiments for Invariances and Data Augmentation for Supervised Music Transcription

Included are 11 notebooks: one for each of the 11 models introduced in the paper. For each model we include a set of weights and optimization statistics that gave us our results. Each notebook is set up to optionally load this set of weights (use the 'init' boolean parameter to the model constructor). This allows you to continue optimization from this initialization, or to re-run the analysis for the numbers reported in the paper.

To run the notebooks, you will need to download the MusicNet dataset. A version of this dataset that has been preprocessed to work with these experiments is available here:

http://homes.cs.washington.edu/~thickstn/icassp_data.tar.gz

Please extract this data to the data/ subdirectory.

The official release of MusicNet can be found here:

http://homes.cs.washington.edu/~thickstn/start.html

Pre-trained models containing the experimental results reported in this paper can be downloaded here:

http://homes.cs.washington.edu/~thickstn/icassp_weights.tar.gz

Please extract this data to the weights/ subdirectory.
