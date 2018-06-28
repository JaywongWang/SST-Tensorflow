"""
Build graph for both training and inference
"""

import tensorflow as tf

class ProposalModel(object):

    def __init__(self, options):
        self.options = options
        self.initializer = tf.random_uniform_initializer(
            minval = - self.options['init_scale'],
            maxval = self.options['init_scale'])

    
    # build inference to get proposal events
    def build_proposal_inference(self, reuse=False):
        inputs = {}
        outputs = {}

        ## dim1: batch, dim2: video sequence length, dim3: video feature dimension
        ## video feature sequence
        video_feat = tf.placeholder(tf.float32, [None, None, self.options['video_feat_dim']], name='video_feat')
        inputs['video_feat'] = video_feat

        ## video feature masking, enable variable-length feature sequence input
        video_feat_mask = tf.placeholder(tf.float32, [None, None], name='video_feat_mask')
        inputs['video_feat_mask'] = video_feat_mask


        batch_size = tf.shape(video_feat)[0]
        
        # set rnn type
        def get_rnn_cell():
            if self.options['rnn_type'] == 'lstm':
                rnn_cell_video = tf.contrib.rnn.LSTMCell(
                    num_units=self.options['rnn_size'],
                    state_is_tuple=True, 
                    initializer=tf.orthogonal_initializer()
                )
            elif self.options['rnn_type'] == 'gru':
                rnn_cell_video = tf.contrib.rnn.GRUCell(
                    num_units=self.options['rnn_size']
                )
            else:
                raise ValueError('Unsupported RNN type.')
            
            return rnn_cell_video
        
        if self.options['rnn_type'] == 'lstm':
            multi_rnn_cell_video = tf.contrib.rnn.MultiRNNCell([get_rnn_cell() for _ in range(self.options['num_rnn_layers'])], state_is_tuple=True)
        elif self.options['rnn_type'] == 'gru':
            multi_rnn_cell_video = tf.contrib.rnn.MultiRNNCell([get_rnn_cell() for _ in range(self.options['num_rnn_layers'])])
        else:
            raise ValueError('Unsupported RNN type.')

        with tf.variable_scope('proposal_module', reuse=reuse) as proposal_scope:
            # video feature sequence encoding: use multi-layer LSTM
            with tf.variable_scope('video_encoder', reuse=reuse) as scope:
                sequence_length = tf.reduce_sum(video_feat_mask, axis=-1)
                initial_state = multi_rnn_cell_video.zero_state(batch_size=batch_size, dtype=tf.float32)
                rnn_outputs, _ = tf.nn.dynamic_rnn(
                    cell=multi_rnn_cell_video, 
                    inputs=video_feat, 
                    sequence_length=sequence_length, 
                    initial_state=initial_state,
                    dtype=tf.float32
                )
                
            rnn_outputs_reshape = tf.reshape(rnn_outputs, [-1, self.options['rnn_size']], name='rnn_outputs_reshape')

            # predict proposal at each time step: use fully connected layer to output scores for every anchors
            with tf.variable_scope('predict_proposal', reuse=reuse) as scope:
                logit_output = tf.contrib.layers.fully_connected(
                    inputs = rnn_outputs_reshape, 
                    num_outputs = self.options['num_anchors'],
                    activation_fn = None
                )

        # score
        proposal_score = tf.sigmoid(logit_output, name='proposal_score')

        proposal_score = tf.reshape(proposal_score, [batch_size, -1, self.options['num_anchors']])

        # outputs from proposal module
        outputs['proposal_score'] = proposal_score

        return inputs, outputs 


    def build_train(self):
        
        inputs = {}
        outputs = {}

        ## dim1: batch, dim2: video sequence length, dim3: video feature dimension
        ## video feature sequence
        video_feat = tf.placeholder(tf.float32, [None, None, self.options['video_feat_dim']], name='video_feat')
        inputs['video_feat'] = video_feat

        ## video feature masking, enable variable-length feature sequence input
        video_feat_mask = tf.placeholder(tf.float32, [None, None], name='video_feat_mask')
        inputs['video_feat_mask'] = video_feat_mask

        ## proposal data, densely annotated
        proposal = tf.placeholder(tf.int32, [None, None, self.options['num_anchors']], name='proposal')
        inputs['proposal'] = proposal

        ## weighting for positive/negative labels (solve imblance data problem)
        proposal_weight = tf.placeholder(tf.float32, [self.options['num_anchors'], 2], name='proposal_weight')
        inputs['proposal_weight'] = proposal_weight

        # get batch size, which is a scalar tensor
        batch_size = tf.shape(video_feat)[0]
        
        if self.options['rnn_drop'] > 0:
            print('using dropout in rnn!')
        
        # set rnn drop out
        rnn_drop = tf.placeholder(tf.float32)
        inputs['rnn_drop'] = rnn_drop
        
        def get_rnn_cell():
            if self.options['rnn_type'] == 'lstm':
                rnn_cell_video = tf.contrib.rnn.LSTMCell(
                    num_units=self.options['rnn_size'],
                    state_is_tuple=True, 
                    initializer=tf.orthogonal_initializer()
                )
            elif self.options['rnn_type'] == 'gru':
                rnn_cell_video = tf.contrib.rnn.GRUCell(
                    num_units=self.options['rnn_size']
                )
            else:
                raise ValueError('Unsupported RNN type.')
            
            rnn_cell_video = tf.contrib.rnn.DropoutWrapper(
                rnn_cell_video,
                input_keep_prob=1.0 - rnn_drop,
                output_keep_prob=1.0 - rnn_drop
            )
            return rnn_cell_video

        if self.options['rnn_type'] == 'lstm':
            multi_rnn_cell_video = tf.contrib.rnn.MultiRNNCell([get_rnn_cell() for _ in range(self.options['num_rnn_layers'])], state_is_tuple=True)
        elif self.options['rnn_type'] == 'gru':
            multi_rnn_cell_video = tf.contrib.rnn.MultiRNNCell([get_rnn_cell() for _ in range(self.options['num_rnn_layers'])])
        else:
            raise ValueError('Unsupported RNN type.')

        with tf.variable_scope('proposal_module') as proposal_scope:
            # video feature sequence encoding: use multi-layer LSTM
            with tf.variable_scope('video_encoder') as scope:
                sequence_length = tf.reduce_sum(video_feat_mask, axis=-1)
                initial_state = multi_rnn_cell_video.zero_state(batch_size=batch_size, dtype=tf.float32)
                rnn_outputs, _ = tf.nn.dynamic_rnn(
                    cell=multi_rnn_cell_video, 
                    inputs=video_feat, 
                    sequence_length=sequence_length, 
                    initial_state=initial_state,
                    dtype=tf.float32
                )
                
            rnn_outputs_reshape = tf.reshape(rnn_outputs, [-1, self.options['rnn_size']], name='rnn_outputs_reshape')

            # predict proposal at each time step: use fully connected layer to output scores for every anchors
            with tf.variable_scope('predict_proposal') as scope:
                logit_output = tf.contrib.layers.fully_connected(
                    inputs = rnn_outputs_reshape, 
                    num_outputs = self.options['num_anchors'],
                    activation_fn = None
                )


        # calculate multi-label loss: use weighted binary cross entropy objective
        proposal_reshape = tf.reshape(proposal, [-1, self.options['num_anchors']], name='proposal_reshape')
        proposal_float = tf.to_float(proposal_reshape)

        # weighting positive samples
        weight0 = tf.reshape(proposal_weight[:, 0], [-1, self.options['num_anchors']])
        # weighting negative samples
        weight1 = tf.reshape(proposal_weight[:, 1], [-1, self.options['num_anchors']])

        # tile weight batch_size times
        weight0 = tf.tile(weight0, [tf.shape(logit_output)[0], 1])
        weight1 = tf.tile(weight1, [tf.shape(logit_output)[0], 1])

        # get weighted sigmoid xentropy loss
        loss_term = tf.nn.weighted_cross_entropy_with_logits(targets=proposal_float, logits=logit_output, pos_weight=weight0)

        loss_term_sum = tf.reduce_sum(loss_term, axis=-1, name='loss_term_sum')

        video_feat_mask = tf.to_float(tf.reshape(video_feat_mask, [-1]))
        proposal_loss = tf.reduce_sum(tf.multiply(loss_term_sum, video_feat_mask)) / tf.reduce_sum(video_feat_mask)

        # summary data, for visualization using Tensorboard
        tf.summary.scalar('proposal_loss', proposal_loss)

        # outputs from proposal module
        outputs['loss'] = proposal_loss

        # L2 regularization
        reg_loss = tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables()])
        outputs['reg_loss'] = reg_loss

        return inputs, outputs


