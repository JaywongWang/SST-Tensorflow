"""
Generate proposals

Suggestion: use non-maximum threshold of 0.8, I found it works almost the best
"""
import os
import numpy as np
import json
import h5py
import argparse
import tensorflow as tf
from opt import *
from data_provider import *
from model import *


def getKey(item):
    return item['score']

"""
Non-Maximum Suppression

I only changes input type to python list (original is Panda table)
"""
def nms_detections(proposals, overlap=0.7):
    """Non-maximum suppression: Greedily select high-scoring detections and
    skip detections that are significantly covered by a previously selected
    detection. This version is translated from Matlab code by Tomasz
    Malisiewicz, who sped up Pedro Felzenszwalb's code.

    Parameters
    ----------
    proposals: list of item, each item is a dict containing 'timestamp' and 'score' field

    Returns
    -------
    new proposals with only the proposals selected after non-maximum suppression.
    """

    if len(proposals) == 0:
        return proposals

    props = np.array([item['timestamp'] for item in proposals])
    scores = np.array([item['score'] for item in proposals])
    t1 = props[:, 0]
    t2 = props[:, 1]
    ind = np.argsort(scores)
    area = (t2 - t1 + 1).astype(float)
    pick = []
    while len(ind) > 0:
        i = ind[-1]
        pick.append(i)
        ind = ind[:-1]
        tt1 = np.maximum(t1[i], t1[ind])
        tt2 = np.minimum(t2[i], t2[ind])
        wh = np.maximum(0., tt2 - tt1 + 1.0)
        o = wh / (area[i] + area[ind] - wh)
        ind = ind[np.nonzero(o <= overlap)[0]]
    nms_props, nms_scores = props[pick, :], scores[pick]

    out_proposals = []
    for idx in range(nms_props.shape[0]):
        prop = nms_props[idx].tolist()
        score = float(nms_scores[idx])
        out_proposals.append({'timestamp': prop, 'score': score})


    return out_proposals


def test(options):
    '''
    Device setting
    '''
    sess_config = tf.ConfigProto()
    sess_config.gpu_options.allow_growth=True
    os.environ['CUDA_VISIBLE_DEVICES'] = str(options['gpu_id'])
    sess = tf.InteractiveSession(config=sess_config)

    # build model
    print('Building model ...')
    model = ProposalModel(options)
    inputs, outputs = model.build_proposal_inference()

    print('Loading data ...')
    data_provision = DataProvision(options)

    print('Restoring model from %s'%options['init_from'])
    saver = tf.train.Saver()
    saver.restore(sess, options['init_from'])

    batch_size = 1 # to loop over all elements
    split = 'test'

    test_batch_generator = data_provision.iterate_batch(split, batch_size)
    video_ids = data_provision.get_ids(split)
    anchors = data_provision.get_anchors()
    localizations = data_provision.get_localization()

    c3d_resolution = options['c3d_resolution']

    print('Start to predict ...')

    count = 0
    

    # output data, for evaluation
    out_data = {}
    out_data['results'] = {}
    results = {}
    proposal_numbers = []
    for batch_data in test_batch_generator:
        proposal_score = sess.run(outputs['proposal_score'], feed_dict={inputs['video_feat']: batch_data['video_feat'], inputs['video_feat_mask']: batch_data['video_feat_mask']})
        
        for sample_id in range(batch_size):
            vid = video_ids[count]
            duration = localizations[split][vid]['duration']
            frame_num = localizations[split][vid]['frame_num']
            gap = (round(0.5*c3d_resolution)/float(frame_num)) * duration

            feat_len = batch_data['video_feat'][sample_id].shape[0]
            print('%d-th video: %s, feat_len: %d'%(count, vid, feat_len))
            
            result = []
            for i in range(feat_len):
                for j in range(options['num_anchors']):
                    # calculate time stamp from feature id
                    end_frame_id = round((i+0.5)*c3d_resolution)
                    start_frame_id = end_frame_id - anchors[j] + 1
                    end_time = (end_frame_id/float(frame_num)) * duration
                    start_time = (start_frame_id/float(frame_num)) * duration
                    
                    if start_time >= 0.-gap:
                        start_time = max(0., start_time)
                        result.append({'timestamp': [start_time, end_time], 'score': float(proposal_score[sample_id, i, j])})

            # add the largest proposal
            result.append({'timestamp': [0., duration], 'score': 1.0})

            print('Number of proposals (before post-processing): %d'%len(result))
    
            # non-maximum suppresion
            print('Non-maximum Suppresion ...')
            result = nms_detections(result, overlap=options['nms_threshold'])

            print('Number of proposals (after nms): %d'%len(result))

            result = sorted(result, key=getKey, reverse=True)

            # score threshold
            result = [item for item in result if item['score'] >= options['proposal_score_threshold']]
            
            print('Number of proposals (after score threshold): %d'%len(result))
            

            results[vid] = result

            proposal_numbers.append(len(result))

            count = count + 1


    out_data['results'] = results

    avg_proposal_num = sum(proposal_numbers) / float(len(proposal_numbers))
    print('Average proposal number: %f'%avg_proposal_num)

    out_json_file = options['out_json_file']

    rootfolder1 = os.path.dirname(out_json_file)
    rootfolder2 = os.path.dirname(rootfolder1)

    if not os.path.exists(rootfolder2):
        print('Make directory %s ...'%rootfolder2)
        os.mkdir(rootfolder2)
    if not os.path.exists(rootfolder1):
        print('Make directory %s ...'%rootfolder1)
        os.mkdir(rootfolder1)


    print('Writing result json file ...')
    with open(out_json_file, 'w') as fid:
        json.dump(out_data, fid)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    options = default_options()
    for key, value in options.items():
        parser.add_argument('--%s'%key, dest=key, type=type(value), default=None)
    args = parser.parse_args()
    args = vars(args)
    for key, value in args.items():
        if value:
            options[key] = value

    test(options)