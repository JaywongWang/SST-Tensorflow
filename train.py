"""
Train your model
"""

import os
import argparse
import numpy as np
import tensorflow as tf
from opt import *
from data_provider import *
from model import * 


def getKey(item):
    return item['score']


# loss evaluation
def evaluation(options, data_provision, sess, inputs, t_loss, t_summary):
    val_loss_list = []
    
    eval_batch_size = options['eval_batch_size']
    val_count = (data_provision.get_size('val')/eval_batch_size)*eval_batch_size
    
    count = 0
    for batch_data in data_provision.iterate_batch('val', eval_batch_size):
        print('Evaluating batch: #%d'%count)
        count += 1
        feed_dict = {inputs['rnn_drop']:0.}
        for key, value in batch_data.items():
            if key not in inputs:
                continue
            feed_dict[inputs[key]] = value

        loss, summary = sess.run(
                    [t_loss, t_summary],
                    feed_dict=feed_dict)
        val_loss_list.append(loss*eval_batch_size)
        
        
    ave_val_loss = sum(val_loss_list) / float(val_count)
    
    return ave_val_loss


def train(options):
    
    '''
    Device setting
    '''
    sess_config = tf.ConfigProto()
    sess_config.gpu_options.allow_growth=True
    os.environ['CUDA_VISIBLE_DEVICES'] = str(options['gpu_id'])
    sess = tf.InteractiveSession(config=sess_config)

    print('Load data ...')
    data_provision = DataProvision(options)

    batch_size = options['batch_size']
    max_epochs = options['max_epochs']
    init_epoch = options['init_epoch']
    lr_init = options['learning_rate']
    lr = lr_init

    n_iters_per_epoch = data_provision.get_size('train') // batch_size
    eval_in_iters = int(n_iters_per_epoch / float(options['n_eval_per_epoch']))

    
    # build model 
    model = ProposalModel(options)
    print('Build model for training stage ...')
    inputs, outputs = model.build_train()
    t_loss = outputs['loss']
    t_reg_loss = outputs['reg_loss']
    print('Building model for validation stage ...')
    i_inputs, i_outputs = model.build_proposal_inference(reuse=True)
    
    # for tensorboard visualization
    t_summary = tf.summary.merge_all()
    t_lr = tf.placeholder(tf.float32)

    
    if options['solver'] == 'adam':
        optimizer = tf.train.AdamOptimizer(learning_rate=t_lr)
    elif options['solver'] == 'sgd':
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=t_lr)
    elif options['solver'] == 'momentum':
        optimizer = tf.train.MomentumOptimizer(learning_rate=t_lr, momentum=options['momentum'])
    elif options['solver'] == 'adadelta':
        optimizer = tf.train.AdadeltaOptimizer(learning_rate=t_lr)
    else:
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=t_lr)


    # gradient clipping option
    if options['clip_gradient_norm'] < 0:
        train_op = optimizer.minimize(t_loss + options['reg'] * t_reg_loss)
    else:
        gvs = optimizer.compute_gradients(t_loss + options['reg'] * t_reg_loss)
        clip_grad_var = [(tf.clip_by_norm(grad, options['clip_gradient_norm']), var) for grad, var in gvs]
        train_op = optimizer.apply_gradients(clip_grad_var)

    # save summary data
    train_summary_writer = tf.summary.FileWriter(options['ckpt_prefix'], sess.graph)

    # initialize all variables
    tf.global_variables_initializer().run()

    # for saving and restoring checkpoints during training
    saver = tf.train.Saver(max_to_keep=100, write_version=1)

    # initialize model from a given checkpoint path
    if options['init_from']:
        print('Init model from %s'%options['init_from'])
        saver.restore(sess, options['init_from'])


    if options['eval_init']:
        print('Evaluating the initialized model ...')
        val_loss = evaluation(options, data_provision, sess, inputs, t_loss, t_summary)
        print('loss: %.4f'%val_loss)
            

    t0 = time.time()
    eval_id = 0
    train_batch_generator = data_provision.iterate_batch('train', batch_size)
    total_iter = 0
    for epoch in range(init_epoch, max_epochs):
        print('epoch: %d/%d, lr: %.1E (%.1E)'%(epoch, max_epochs, lr, lr_init))
        for iter in range(n_iters_per_epoch):
            batch_data = next(train_batch_generator)
            feed_dict = {
                t_lr: lr,
                inputs['rnn_drop']: options['rnn_drop']
            }
            for key, value in batch_data.items():
                if key not in inputs:
                    continue
                feed_dict[inputs[key]] = value
            _, summary, loss, reg_loss = sess.run(
                        [train_op, t_summary, t_loss, t_reg_loss],
                        feed_dict=feed_dict)
            
            if iter == 0 and epoch == init_epoch:
                smooth_loss = loss
            else:
                smooth_loss = 0.9 * smooth_loss + 0.1 * loss
            
            if iter % options['n_iters_display'] == 0:
                print('iter: %d, epoch: %d/%d, \nlr: %.1E, loss: %.4f, reg_loss: %.4f'%(iter, epoch, max_epochs, lr, loss, reg_loss))
                train_summary_writer.add_summary(summary, iter + epoch * n_iters_per_epoch)
            
            if (total_iter+1) % eval_in_iters == 0:
                
                print('Evaluating model ...')
                val_loss = evaluation(options, data_provision, sess, inputs, t_loss, t_summary)
                print('loss: %.4f'%val_loss)  

                checkpoint_path = '%sepoch%02d_%.2f_%02d_lr%f.ckpt' % (options['ckpt_prefix'], epoch, val_loss, eval_id, lr)

                saver.save(sess, checkpoint_path)
                
                eval_id = eval_id + 1

            total_iter += 1
                
                

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

    work_dir = options['ckpt_prefix']
    if not os.path.exists(work_dir) :
        os.makedirs(work_dir)

    train(options)

