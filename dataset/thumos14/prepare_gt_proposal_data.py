'''
data format: vid1: [{'timestamp': [t0, t1]}, {'timestamp': [t2, t3]}], vid2: [], ...

"valid frame number" is number of those frames that are used to calculate C3D features (e.g., a video with 1606 frames have 1600 valid frames since the final 6 frames are not used to calcualte C3D).
"total frame number" is number of all frames in a video.


Please extract video frames and obtain corresponding meta video data in your side.
Please also extract C3D features for all videos and store them in hdf5 file.
'''

import os
import json
import random
import h5py

val_anno_folder = 'th14_temporal_annotations_validation/annotation'
test_anno_folder = 'th14_temporal_annotations_test/annotation'


video_info_file = 'thumos_video_info.json' # THUMOS14 website provided meta data, I already convert original .mat data to .json data
video_info = json.load(open(video_info_file, 'r'))

sources = {'val': val_anno_folder, 'test': test_anno_folder}


val_split = {'train': 0.8, 'val': 0.2}  # further split validation set into train set and validation set

# feature file
feat_path = 'features/thumos14_c3d_fc6.hdf5'
feat_data = h5py.File(feat_path, 'r')


feat_resolution = 16

for split in sources.keys():
	print('Processing split: %s ...'%split)
	out_data = dict()
	source = sources[split]
	files = os.listdir(source)
	files.remove('Ambiguous_%s.txt'%split)

	for file in files:
		filepath = os.path.join(source, file)
		print('Reading file: %s from %s'%(file, source))
		annos = open(filepath, 'r').readlines()
		for line in annos:
			items = line.strip().split()
			assert len(items) == 3
			vid = items[0]

			extracted_frame_num = int(feat_data[vid]['total_frames'].value)
			valid_frame_num = int(feat_data[vid]['valid_frames'].value)
			start_time, end_time = float(items[1]), float(items[2])
			start_frame = int(extracted_frame_num*(start_time/video_info[vid]['duration']))
			end_frame = int(extracted_frame_num*(end_time/video_info[vid]['duration']))
			start_feat = start_frame//feat_resolution
			end_feat = end_frame//feat_resolution

			if vid in out_data.keys():
				out_data[vid]['timestamps'].append([start_time, end_time])
				out_data[vid]['framestamps'].append([start_frame, end_frame])
				out_data[vid]['featstamps'].append([start_feat, end_feat])
			else:
				out_data[vid] = {}
				out_data[vid]['timestamps'] = [[start_time, end_time]]
				out_data[vid]['framestamps'] = [[start_frame, end_frame]]
				out_data[vid]['featstamps'] = [[start_feat, end_feat]]

				assert vid in video_info.keys()
				out_data[vid]['duration'] = video_info[vid]['duration']
				out_data[vid]['frame_num'] = video_info[vid]['frame_num']
				out_data[vid]['frame_rate'] = video_info[vid]['frame_rate']

				out_data[vid]['extracted_frame_num'] = extracted_frame_num
				out_data[vid]['valid_frame_num'] = valid_frame_num

	
	if split == 'val':
		k = out_data.keys()
		N = len(k)
		train_num = int(val_split['train']*N)
		val_num = N - train_num
		random.shuffle(k)
		train_keys = k[:train_num]
		val_keys = k[train_num:]

		out_data_train = {k:out_data[k] for k in train_keys}
		out_data_val = {k:out_data[k] for k in val_keys}
		print('Writing train json data ...')
		with open('gt_proposals/thumos14_temporal_proposal_train.json', 'w') as fid:
			json.dump(out_data_train, fid)

		print('Writing val json data ...')
		with open('gt_proposals/thumos14_temporal_proposal_val.json', 'w') as fid:
			json.dump(out_data_val, fid)

	else:
		print('Writing %s json data ...'%split)
		with open('gt_proposals/thumos14_temporal_proposal_%s.json'%split, 'w') as fid:
			json.dump(out_data, fid)

print('DONE.')

