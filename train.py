import os, sys
import datetime
from model import SequenceClassifier
from util import create_tfrecords, read_bashdata, make_example, parse_example
import tensorflow as tf
import numpy as np

BATCH_SIZE = 128
MAX_ITER = 100000
N_EPOCHS = 100
LEARNING_RATE = 0.005


def main(save_path):
    with tf.name_scope('Inputs'):
        # Create TFRecords file if doesn't exist
        records_path = '.'
        filename = os.path.join(records_path, 'bash_data.TFRecords')
        if not os.path.isfile(filename):
            create_tfrecords(records_path=records_path, shuffle=True, 
                             single_out_user=7, min_length=2)
        
        # Set log directory to resume from previous path                 
        if save_path:
            log_dir, _ = os.path.split(save_path)
        else:
            log_dir = os.path.abspath('./logdir/' + str(datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")))
            
        # Queue examples
        filename_queue = tf.train.string_input_producer([filename]*N_EPOCHS)
        reader = tf.TFRecordReader()
        _, example = reader.read(filename_queue)
        sequence_parsed, context_parsed = parse_example(example)
        
        tokens = sequence_parsed['tokens']
        labels = sequence_parsed['labels']
        length = context_parsed['length']
        
        
    # Session
    sess = tf.Session()
    
    # Build Model
    keep_prob = tf.placeholder(tf.float64)
    model = SequenceClassifier(
        tokens, 
        labels, 
        length, 
        batch_size=BATCH_SIZE,
        n_cells=64,
        n_hidden=32,
        n_stacks=1,
        partial_classification=False,
        embeddings=np.load('embeddings.npy'),
        )
        
    optimizer = tf.train.AdamOptimizer(LEARNING_RATE)
    global_step = tf.Variable(0, name='global_step', trainable=False)
    learning_rate = tf.train.exponential_decay(LEARNING_RATE,
                                 global_step,
                                 200,
                                 0.9,
                                 staircase=True)
    train_op = optimizer.minimize(model.loss, global_step=global_step)
    
    
    # Saver
    saver = tf.train.Saver()
    
    # Coordinator
    coord = tf.train.Coordinator()
    
    # Summary
    merged = tf.merge_all_summaries()
    writer = tf.train.SummaryWriter(log_dir, sess.graph)
    
    with tf.name_scope('Training'):
        try:
            if save_path:
                print "Restoring Model from ", save_path
                saver.restore(sess, os.path.join('file://', os.path.abspath(save_path)))
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
                feed_dict = {
                    keep_prob: 0.5,
                    }
                lr, p,y, embeddings, l,a,_ = sess.run((
                                    learning_rate, 
                                    model.probs, 
                                    model.batched_labels,
                                    model.embeddings,
                                    model.loss, 
                                    model.acc, 
                                    train_op), 
                                    feed_dict=feed_dict)
    
                mer = sess.run(merged, feed_dict=feed_dict)
                writer.add_summary(mer, step)
                if step % 1 == 0:
                    print 'step: %4d' % step, 'Batch Loss: %3.5f' % l, ' Batch Acc: %01.2f' % a, 'LR: %0.6f' % lr
                    
                if step % 50 == 0:
                    saver.save(sess, os.path.join(log_dir, 'model%d.ckpt' % step))
                    print 'saving embeddings'
                    np.save("embeddings.npy", embeddings)
                    print 'probs:', p[0:5]
                    print 'predictions:', np.argmax(p[0:32], axis=1)
                    print 'actual:', y.reshape(-1)[0:32]
                
        except KeyboardInterrupt:
            print()
        finally:
            saver.save(sess, os.path.join(log_dir, 'model%d.ckpt' % step))
            coord.request_stop()
    
    
if __name__ == '__main__':
    save_path = None
    if len(sys.argv) > 1:
        save_path = sys.argv[1]
    main(save_path)