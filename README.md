# SST-Tensorflow

Tensorflow Implementation of the Paper [SST: Single-Stream Temporal Action Proposals](http://vision.stanford.edu/pdf/buch2017cvpr.pdf) in *CVPR* 2017.


### Data Preparation

I run experiments on *THUMOS14* dataset.

Please download video data and annotation data from the website [THUMOS14](http://crcv.ucf.edu/THUMOS14/download.html).

Extract C3D features for non-overlap 16-frame snippets from the 412 videos (212 *val* videos + 200 *test* videos, I found one *val* video missing) for the task of temporal action proposals. Put them in *dataset/thumos14/features/*.

I use *fc6* features in my experiment.

Please follow the script *dataset/thumos14/prepare_gt_proposal_data.py* to generate ground-truth proposal data for train/val/test split.

After that, please generate anchor weights (for handling imbalance class problem) by uniformly sampling video streams (follow *dataset/thumos14/anchors/get_anchor_weight.py*) or just use my pre-calculated weights (*weights.json*).


### Hyper Parameters

I provide the best configuration (from my experiments) in *opt.py*, including model setup, training options, and testing options.

### Training

Train your model using the script *train.py*. Run around 50 epochs and pick the best checkpoint (with the smallest val loss) for prediction.

### Prediction

Follow the script *test.py* to make proposal predictions.

### Evaluation

Follow the script *eval.py* to evaluate your proposal predictions.

### Results

My implemented results can be found in *results/1*. They are comparable or even better than the reported ones.

<table>
  <tr>
    <th>Method</th>
    <th>Recall@1000 at tIoU=0.8</th>
  </tr>
  <tr>
    <th>SST (paper)</th>
    <th>0.672</th>
  </tr>
  <tr>
    <th>SST (my impl)</th>
    <th>0.696</th>
  </tr>
</table>

![alt text](results/1/sst_recall_vs_proposal.png "Average Recall vs Average Proposal Number")

![alt text](results/1/sst_recall_vs_tiou.png "Recall@1000 vs tIoU")

### Dependencies

tensorflow==1.0.1

python==2.7.5

Other versions may also work.

### Acknowledgements

Great thanks to Shyamal Buch for really helpful discussion.
