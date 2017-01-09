import argparse
import os, sys
from model import SequenceClassifier
from util import parse_example
import tensorflow as tf

BATCH_SIZE = 128
DATA_FILE = 'bash_data_test.TFRecords'

def get_arguments():
    parser = argparse.ArgumentParser(description='Bash model training script')
    parser.add_argument('--data_file', type=str, default=DATA_FILE,
                        help='The TFRecords data file')
    parser.add_argument('--model_file', type=str, default=None,
                        help='Checkpoint for the model you wish to test')
    parser.add_argument('--n_labels', type=int, default=9,
                        help='Number of labels in dataset.')   
    parser.add_argument('--min_length', type=int, default=2,
                        help='Train against (sub)sequences with '
                        'length >min_length')          
    return parser.parse_args()
    
def test(data_file, model_file, n_labels, min_length):
    with tf.name_scope('Inputs'):
        # Queue examples
        filename_queue = tf.train.string_input_producer([data_file], num_epochs=1, capacity=256)
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
        )
        
    # Saver
    saver = tf.train.Saver()
    
    # Coordinator
    coord = tf.train.Coordinator()
    
    
    init_op = tf.initialize_all_variables()
    init_local = tf.initialize_local_variables()
    sess.run([init_local, init_op])
    
    print "Restoring Model from ", model_file
    saver.restore(sess, 
        os.path.join('file://', os.path.abspath(model_file)))
    tf.train.start_queue_runners(sess=sess, coord=coord)
    
    try:
        c = 0
        total_loss = 0
        total_acc = 0
        while True:
            loss, acc, n = sess.run((model.loss, model.acc, reader.num_records_produced()))
            c += 1
            total_loss += loss
            total_acc += acc
    except tf.errors.OutOfRangeError, e:
        coord.request_stop(e)
    finally:
        coord.request_stop()
        print 'Test Loss: %f' % (total_loss/c)
        print 'Test Accuracy: %f' % (total_acc/c)
    

def main():
    args = get_arguments()
    
    test(args.data_file, args.model_file, args.n_labels, args.min_length)
    
if __name__ == '__main__':
    main()