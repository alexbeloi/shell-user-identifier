import argparse
import os, sys
import datetime
from model import SequenceClassifier
from util import parse_example
import tensorflow as tf
import numpy as np

BATCH_SIZE = 128
DATA_FILE = 'bash_data_test.TFRecords'
MAX_ITER = 100000
N_EPOCHS = 1000
LEARNING_RATE = 0.01

def get_arguments():
    parser = argparse.ArgumentParser(description='Bash model training script')
    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE,
                        help='How many wav files to process at once.')
    parser.add_argument('--data_file', type=str, default=DATA_FILE,
                        help='The TFRecords data file')
    parser.add_argument('--restore_from', type=str, default=None,
                        help='Directory in which to restore the model from.')
    parser.add_argument('--learning_rate', type=float, default=LEARNING_RATE,
                        help='Learning rate for training.')
    parser.add_argument('--n_labels', type=int, default=9,
                        help='Number of labels in dataset.')
    parser.add_argument('--min_length', type=int, default=2,
                        help='Train against (sub)sequences with '
                        'length >min_length')
    parser.add_argument('--l2', type=float, default=0.01,
                        help='L2 regularization beta parameter')            
    return parser.parse_args()

def train(data_file, 
          model_file=None,
          log_dir=None, 
          learning_rate=LEARNING_RATE, 
          batch_size=BATCH_SIZE,
          n_labels=10,
          min_length=2,
          l2_reg=0.01,):
    with tf.name_scope('Inputs'):
        # Queue examples
        filename_queue = tf.train.string_input_producer([data_file],
                                                        num_epochs=N_EPOCHS, 
                                                        capacity=batch_size*2)
        reader = tf.TFRecordReader()
        _, example = reader.read(filename_queue)
        sequence_parsed, context_parsed = parse_example(example)
        
        tokens = sequence_parsed['tokens']
        labels = sequence_parsed['labels']
        length = context_parsed['length']
        
    # Session
    sess = tf.Session()
    
    # Build Model
    model = SequenceClassifier(
        tokens, 
        labels, 
        length, 
        batch_size=BATCH_SIZE,
        n_cells=64,
        n_hidden=32,
        n_stacks=1,
        n_labels=n_labels+1,
        partial_classification=True,
        min_length=min_length,
        l2_reg=l2_reg,
        )
    
    global_step = tf.Variable(0, name='global_step', trainable=False)
    learning_rate = tf.train.exponential_decay(LEARNING_RATE,
                                 global_step,
                                 100,
                                 0.9,
                                 staircase=True)
    optimizer = tf.train.AdamOptimizer(learning_rate)
    train_op = optimizer.minimize(model.loss, global_step=global_step)
    
    # Saver
    saver = tf.train.Saver(max_to_keep=10)
    
    # Coordinator
    coord = tf.train.Coordinator()
    
    # Summary
    merged = tf.merge_all_summaries()
    writer = tf.train.SummaryWriter(log_dir, sess.graph)
    
    with tf.name_scope('Training'):
        try:
            if model_file:
                print "Restoring Model from ", model_file
                saver.restore(sess, 
                    os.path.join('file://', os.path.abspath(model_file)))
            else:
                print "Initializing model variables"
                init = tf.initialize_all_variables()
                sess.run(init)
            init_local = tf.initialize_local_variables()
            sess.run(init_local)
            start_step = sess.run(global_step)
            writer.add_session_log(tf.SessionLog(status=tf.SessionLog.START), 
                global_step=start_step)
            tf.train.start_queue_runners(sess=sess, coord=coord)
            
            # Training Loop
            for step in range(start_step, MAX_ITER):
                lr, l, a, _ = sess.run((learning_rate, 
                                        model.loss, 
                                        model.acc, 
                                        train_op))
    
                mer = sess.run(merged)
                writer.add_summary(mer, step)
                if step % 1 == 0:
                    print 'step: %4d' % step, \
                          'Batch Loss: %3.5f' % l, \
                          'Batch Acc: %01.2f' % a, \
                          'LR: %0.6f' % lr
                    
                if step % 50 == 0:
                    saver.save(sess, os.path.join(log_dir, 'model%d.ckpt' % step))
                
        except KeyboardInterrupt:
            print()
        finally:
            saver.save(sess, os.path.join(log_dir, 'model%d.ckpt' % step))
            coord.request_stop()
    
def main():
    args = get_arguments()
    
    data_file = os.path.normpath(args.data_file)
    if args.restore_from:
        log_dir, _ = os.path.split(args.restore_from)
        model_file = os.path.normpath(args.restore_from)
    else:
        log_dir = os.path.abspath('./logdir/' + \
            str(datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")))
        model_file = None
    
    train(data_file,
          model_file,
          log_dir, 
          args.learning_rate, 
          args.batch_size,
          args.n_labels,
          args.min_length,
          args.l2)

if __name__ == '__main__':
    main()