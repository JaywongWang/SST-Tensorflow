#-*- coding: utf-8 -*-

"""
Data provider: to provide train/val/test data to the model
"""
import random
import numpy as np
import os
import h5py
from collections import OrderedDict
import json
from opt import *
import random
import math


class DataProvision:
    def __init__(self, options):
        
        self._options = options
        self._splits = {'train', 'val', 'test'}

        self._ids = {}          # video ids: a dictionary containing train/val/test id lists
        self._sizes = {}        # size of train/val/test split data
        self._localization = {} # time stamp data

        self._anchors = list(range(self._options['c3d_resolution'], (self._options['num_anchors'] + 1) * self._options['c3d_resolution'], self._options['c3d_resolution']))  # proposal anchors (in frame number)
        
        print('Data Size:')
        for split in self._splits:
            proposal_data = json.load(open(os.path.join(self._options['proposal_data_path'], 'thumos14_temporal_proposal_%s.json'%split), 'r'))
            tmp_ids = list(proposal_data.keys())
            self._ids[split] = tmp_ids
            self._sizes[split] = len(self._ids[split])
            self._localization[split] = proposal_data

            print('%s-split: %d videos.'%(split, self._sizes[split]))


        # feature dictionary
        print('Loading c3d features ...')
        features = h5py.File(self._options['feature_data_path'], 'r')
        self._feature_ids = features.keys()
        self._features = {video_id:features[video_id]['c3d_features'].value for video_id in self._feature_ids}

        # load label weight data
        print('Loading anchor weight data ...')
        self._proposal_weight = json.load(open(self._options['anchor_weight_path'], 'r'))

        if not self._options['use_weight']:
            self._proposal_weight = np.ones(shape=(self._options['num_anchors'], 2)) / 2.0

        # when using tensorflow built-in function: tf.nn.weighted_cross_entropy_with_logits()
        for i in range(len(self._proposal_weight)):
            self._proposal_weight[i][0] /= self._proposal_weight[i][1]
            self._proposal_weight[i][1] = 1.


        print('Done loading.')


    def get_size(self, split):
        return self._sizes[split]

    def get_ids(self, split):
        return self._ids[split]

    def get_localization(self):
        return self._localization

    def get_anchors(self):
        return self._anchors

    def get_iou(self, pred, gt):
        start_pred, end_pred = pred
        start, end = gt
        intersection = max(0, min(end, end_pred) - max(start, start_pred))
        union = min(max(end, end_pred) - min(start, start_pred), end-start + end_pred-start_pred)
        iou = float(intersection) / (union + 1e-8)

        return iou

    def get_intersection(self, region1, region2):
        start1, end1 = region1
        start2, end2 = region2
        start = max(start1, start2)
        end = min(end1, end2)

        return (start, end)


    # process batch data: to generate formatted tensor and necessary mask
    def process_batch_data(self, batch_data):

        data_length = []
        for data in batch_data:
            data_length.append(data.shape[0])
        max_length = max(data_length)

        dim = batch_data[0].shape[1]

        out_batch_data = np.zeros(shape=(len(batch_data), max_length, dim), dtype='float32')
        out_batch_data_mask = np.zeros(shape=(len(batch_data), max_length), dtype='int32')

        for i, data in enumerate(batch_data):
            effective_len = data.shape[0]
            out_batch_data[i, :effective_len, :] = data
            out_batch_data_mask[i, :effective_len] = 1

        out_batch_data = np.asarray(out_batch_data, dtype='float32')
        out_batch_data_mask = np.asarray(out_batch_data_mask, dtype='int32')

        return out_batch_data, out_batch_data_mask


    # generate train/val/test data
    def iterate_batch(self, split, batch_size):

        ids = list(self._ids[split])

        if split == 'train':
            print('Randomly shuffle training data ...')
            random.shuffle(ids)

        current = 0
        
        
        while True:
            batch_feature = []
            batch_proposal = []

            c3d_resolution = self._options['c3d_resolution']

            for sample_id in range(batch_size):
                vid = ids[sample_id+current]

                feature = self._features[vid]
                feature_len = feature.shape[0]

                
                # sampling
                if split == 'train':
                    sample_len = self._options['sample_len']
                else:
                    sample_len = feature_len

                # starting feature id relative to original video
                start_feat_id = random.randint(0, max((feature_len-sample_len), 0))
                end_feat_id = min(start_feat_id+sample_len, feature_len)
                feature = feature[start_feat_id:end_feat_id]
                start_frame_id = start_feat_id * c3d_resolution + c3d_resolution / 2
                end_frame_id = (end_feat_id - 1) * c3d_resolution + c3d_resolution / 2

                batch_feature.append(feature)

                # the ground truth proposal and caption should be changed according to the sampled stream
                localization = self._localization[split][vid]
                framestamps = localization['framestamps']

                n_anchors = self._options['num_anchors']

                # generate proposal groud truth data
                gt_proposal = np.zeros(shape=(sample_len, n_anchors), dtype=np.int16)
                for stamp_id, stamp in enumerate(framestamps):
                    start = stamp[0]
                    end = stamp[1]

                    # only need to check whether proposals that have end point at region of (frame_check_start, frame_check_end) are "correct" proposals
                    start_point = max((start + end) / 2, 0)
                    end_point = end + (end - start + 1)
                    frame_check_start, frame_check_end = self.get_intersection((start_point, end_point + 1), (start_frame_id, end_frame_id+1))
                    feat_check_start, feat_check_end = frame_check_start // c3d_resolution, frame_check_end // c3d_resolution

                    for feat_id in range(feat_check_start, feat_check_end + 1):
                        frame_id = feat_id*c3d_resolution + c3d_resolution/2
                        for anchor_id, anchor in enumerate(self._anchors):
                            pred = (frame_id + 1- anchor, frame_id + 1)
                            tiou = self.get_iou(pred, (start, end + 1))
                            
                            if tiou > 0.5:
                                gt_proposal[feat_id-start_feat_id, anchor_id] = 1
                    

                batch_proposal.append(gt_proposal)
            
            # generate tensor numpy data
            batch_feature, batch_feature_mask = self.process_batch_data(batch_feature)
            batch_proposal, _ = self.process_batch_data(batch_proposal)

            # serve as a tuple
            batch_data = {'video_feat': batch_feature, 'proposal': batch_proposal, 'video_feat_mask': batch_feature_mask, 'proposal_weight': np.array(self._proposal_weight)}

            
            yield batch_data

            current = current + batch_size
            
            if current + batch_size > self.get_size(split):
                current = 0
                # at the end of list, shuffle it
                if split == 'train':
                    print('Randomly shuffle training data ...')
                    random.shuffle(ids)
                    print('The new shuffled ids are:')
                    print('%s, %s, %s, ..., %s'%(ids[0], ids[1], ids[2], ids[-1]))
                    
                else:
                    break
