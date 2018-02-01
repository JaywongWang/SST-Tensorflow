# SST-Tensorflow

Tensorflow Implementation of the Paper [SST: Single-Stream Temporal Action Proposals](http://vision.stanford.edu/pdf/buch2017cvpr.pdf) in CVPR 2017.


### Data Preparation

I run experiments on *THUMOS14* dataset.

Please download video data and annotation data from the website [THUMOS14](http://crcv.ucf.edu/THUMOS14/download.html).

Extract C3D features for the 412 videos (212 val videos + 200 test videos, I found *one val video missing*) for the task of temporal action proposals. Put them in *dataset/thumos14/features/*.

I use *fc6* features in my experiment.

Please follow script *dataset/thumos14/prepare_gt_proposal_data.py* to generate ground-truth proposal data for training and evaluating your model.

After that, please generate anchor weights (for handling imbalance samples problem) by uniformly sampling video streams (follow *dataset/thumos14/anchors/get_anchor_weight.py*) or just use my pre-calculated weights (*weights.json*).


### Hyper Parameters

I give the best options (from my experiments) in *opt.py*, including model configeration

### Training

Train your model using the script *train.py*.

### Prediction

Follow the script *test.py* to make proposal predictions.

### Evaluation

Follow the script *eval.py* to evaluate your proposal predictions.

### Results

My implemented results can be found in *results/1*. They are comparable or even better than the reported ones.

### Dependencies
tensorflow==1.0.1
python==2.7.5
Other visions may also work, but not guaranteed.
