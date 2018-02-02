'''
This script is to calculate class weights for Thumos14 dataset by uniform random sampling

Please first extract C3D features for THUMOS14 dataset
'''

import os
import h5py
import json
import math
import numpy as np
import h5py
import random

"""
Calculate tIoU
"""
def get_iou(pred, gt):
    start_pred, end_pred = pred
    start, end = gt
    intersection = max(0, min(end, end_pred) - max(start, start_pred))
    union = min(max(end, end_pred) - min(start, start_pred), end-start + end_pred-start_pred)
    iou = float(intersection) / (union + 1e-8)

    return iou

"""
Calculate intersection of two regions
"""
def get_intersection(region1, region2):
    start1, end1 = region1
    start2, end2 = region2
    start = max(start1, start2)
    end = min(end1, end2)

    return (start, end)



feature_path = '../features/thumos14_c3d_fc6.hdf5'
features = h5py.File(feature_path, 'r')

splits = {'train', 'val', 'test'}

proposal_source = '../gt_proposals'

split = 'train'
proposal_file_path = os.path.join(proposal_source, 'thumos14_temporal_proposal_%s.json'%split)
proposal_train = json.load(open(proposal_file_path, 'r'))

n_anchors = 32
c3d_resolution = 16
# anchor length is measured in frame number 
anchors = list(range(c3d_resolution, (n_anchors+1)*c3d_resolution, c3d_resolution))
sample_len = 2048/16 # length of sampled stream, measured in feature number 
sample_num = 100     # sampling number for a video 

print('Anchors are: (in frames)')
print(anchors)

# count matched samples for each anchor
count_anchors = [0 for _ in range(n_anchors)]

# total sampled length
sum_sample_length = 0

# weighting positive/negative classes for each anchor
weights = [[0., 0.] for _ in range(n_anchors)]

# output path to save calculated weights
out_weight_path = 'weights.json'

print('Get anchor weights ...')
# encode proposal information
split = 'train'
split_data = json.load(open(os.path.join(proposal_source, 'thumos14_temporal_proposal_%s.json'%split)))
video_ids = split_data.keys()

# loop over all training videos
for index, vid in enumerate(video_ids):
    print('Processing video id: %s'%vid)
    data = split_data[vid]
    framestamps = data['framestamps']
    feature = features[vid]['c3d_features'].value
    feature_len = feature.shape[0]
    frame_num = c3d_resolution*feature_len  # valid frame number

    this_sample_len = min(feature_len, sample_len)

    for sample_id in range(sample_num):
        # sample with stride = c3d_resolution
        start_feat_id = random.randint(0, max((feature_len - sample_len), 0))
        end_feat_id = min(start_feat_id + sample_len, feature_len)
        start_frame_id = start_feat_id * c3d_resolution + c3d_resolution / 2
        end_frame_id = (end_feat_id - 1) * c3d_resolution + c3d_resolution / 2

        sum_sample_length += this_sample_len

        for stamp_id, stamp in enumerate(framestamps):
            start = stamp[0]
            end = stamp[1]
            
            # calculate corresponding starting/ending feature ids (where proposal ends) that possibly cover a ground-truth proposal
            start_point = max((start + end) / 2, 0)
            end_point = end + (end - start + 1)
            frame_check_start, frame_check_end = get_intersection((start_point, end_point + 1), (start_frame_id, end_frame_id+1))
            feat_check_start, feat_check_end = frame_check_start / c3d_resolution, frame_check_end / c3d_resolution
            if frame_check_start >= frame_check_end:
                print('sample: #%d: (%d, %d), stamp: #%d: (%d, %d), OUT OF Range'%(sample_id, start_frame_id, end_frame_id, stamp_id, start, end))
                continue
            else:
                print('sample: #%d: (%d, %d), stamp: #%d: (%d, %d), frame_check: (%d, %d)'%(sample_id, start_frame_id, end_frame_id, stamp_id, start, end, frame_check_start, frame_check_end))
            for feat_id in range(feat_check_start, feat_check_end + 1):
                frame_id = feat_id*c3d_resolution + c3d_resolution/2
                for anchor_id, anchor in enumerate(anchors):
                    pred = (frame_id + 1- anchor, frame_id + 1)
                    tiou = get_iou(pred, (start, end + 1))
                    #print('tiou: %f'%tiou)
                    if tiou > 0.5:
                        count_anchors[anchor_id] += 1


for i in range(n_anchors):
    # weight for negative label
    weights[i][1] = count_anchors[i] / float(sum_sample_length)
    # weight for positive label
    weights[i][0] = 1. - weights[i][1]


print('Writing ...')
with open(out_weight_path, 'w') as fid:
    json.dump(weights, fid)

with open('weights.txt', 'w') as fid:
    for w in weights:
        fid.write('%.4f\t%.4f\n'%(w[0], w[1]))


