import tensorflow as tf
import numpy as np

N_CLASSES = 9+1
VOCABULARY_SIZE = 2358 + 1
EMBEDDING_SIZE = 16

class SequenceClassifier():
    def __init__(
            self, 
            inputs,
            labels,
            length,
            batch_size = 128,
            n_cells = 256,
            ):
        
        b_tokens, b_labels, b_lengths = \
            self.batch(inputs, labels, length, batch_size)
        self.logits, self.probs = self.build_model(b_tokens, b_lengths, n_cells)
        self.loss = self.build_loss(self.logits, b_labels, b_lengths)
        self.acc = self.build_accuracy(self.probs, b_labels, b_lengths)
        
    def batch(self, inputs, labels, length, batch_size = 128):
        with tf.variable_scope('Batching'):
            batched_tokens, batched_labels, batched_lengths = tf.train.batch(
                tensors=[inputs, labels, length],
                batch_size=batch_size,
                dynamic_pad=True,
                name='batch',
                allow_smaller_final_batch=True,
            )
        return batched_tokens, batched_labels, batched_lengths
        
    def build_model(self, inputs, lengths, n_cells = 64, n_hidden = 16):
        with tf.variable_scope('Model'):
            embeddings_x = tf.random_uniform([VOCABULARY_SIZE, EMBEDDING_SIZE],
                -1.0, 1.0, dtype=tf.float64, seed=1)
            inputs_embedded = tf.nn.embedding_lookup(embeddings_x, inputs)
                            
            with tf.variable_scope('lstm'):
                cell = tf.nn.rnn_cell.LSTMCell(num_units=n_cells, state_is_tuple=True)
                cell = tf.nn.rnn_cell.DropoutWrapper(cell=cell, output_keep_prob=0.5)
                cell = tf.nn.rnn_cell.MultiRNNCell(cells=[cell] * 4, state_is_tuple=True)

                outputs, states = tf.nn.dynamic_rnn(
                    cell=cell,
                    dtype=tf.float64,
                    sequence_length=lengths,
                    inputs=inputs_embedded,
                )
                  
            W = tf.get_variable('W', initializer = \
                tf.random_normal(
                    [n_cells, N_CLASSES], stddev=0.01, dtype=tf.float64))
            b = tf.get_variable('b', initializer = \
                tf.random_normal([N_CLASSES], stddev=0.01, dtype=tf.float64))
    
            outputs_flat = tf.reshape(outputs, [-1, n_cells])
            logits_flat = tf.batch_matmul(outputs_flat, W) + b
            probs_flat = tf.nn.softmax(logits_flat)
        return logits_flat, probs_flat
        
    def build_loss(self, logits_flat, labels, lengths):
        with tf.variable_scope('Loss'):
            labels_flat = tf.reshape(labels, [-1])
            losses = tf.nn.sparse_softmax_cross_entropy_with_logits(
                        logits_flat, labels_flat)
            mask = tf.sign(tf.to_double(labels_flat))
            masked_losses = mask * losses
            masked_losses = tf.reshape(masked_losses, tf.shape(labels))
            
            loss_by_example = tf.reduce_sum(masked_losses, reduction_indices=1) / tf.to_double(lengths)
            mean_loss = tf.reduce_mean(loss_by_example)
            
        return mean_loss
            
    def build_accuracy(self, probs_flat, labels, lengths):
        with tf.variable_scope('Accuracy'):
            labels_flat = tf.reshape(labels, [-1])
            mask = tf.sign(tf.to_double(labels_flat))
            correct = tf.equal(tf.argmax(probs_flat, 1), labels_flat-1)
            correct = mask * tf.to_double(correct)
            correct = tf.reshape(correct, tf.shape(labels))
            correct_reduced = tf.reduce_sum(correct, reduction_indices=1) / tf.to_double(lengths)
            accuracy = tf.reduce_mean(tf.to_double(correct_reduced), name='accuracy')    
        return accuracy
        