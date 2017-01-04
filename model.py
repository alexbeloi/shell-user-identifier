import tensorflow as tf
import numpy as np

VOCABULARY_SIZE = 2358 + 1
EMBEDDING_SIZE = 32

class SequenceClassifier():
    def __init__(
            self, 
            inputs,
            labels,
            length,
            batch_size=128,
            n_cells=256,
            n_stacks=4,
            n_hidden=32,
            n_labels=10,
            partial_classification=True,
            ):
        
        b_tokens, b_labels, b_lengths = \
            self.batch(inputs, labels, length, batch_size)
        logits, probs = \
            self.build_model(
                            b_tokens,
                            b_lengths,
                            n_cells=n_cells,
                            n_stacks=n_stacks,
                            n_hidden=n_hidden,
                            n_labels=n_labels,
                            )
        self.loss = self.build_loss(
                                    logits,
                                    b_labels,
                                    b_lengths,
                                    partial_classification,
                                    )
        self.acc = self.build_accuracy(probs, b_labels, b_lengths, partial_classification)
        
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
        
    def build_model(self, inputs, lengths, n_cells = 64, n_hidden = 32, n_stacks = 1, n_labels = 10):
        with tf.variable_scope('Model') as vs:
            embeddings = tf.get_variable('embeddings', initializer = \
                tf.random_uniform([VOCABULARY_SIZE, EMBEDDING_SIZE],
                    -1.0, 1.0, dtype=tf.float64, seed=1), trainable=False)
            
            inputs_embedded = tf.nn.embedding_lookup(embeddings, inputs)
            with tf.variable_scope('lstm'):
                cell = tf.nn.rnn_cell.LSTMCell(num_units=n_cells, state_is_tuple=True)
                cell = tf.nn.rnn_cell.DropoutWrapper(cell=cell, output_keep_prob=0.5)
                cell = tf.nn.rnn_cell.MultiRNNCell(cells=[cell] * n_stacks, state_is_tuple=True)

                outputs, states = tf.nn.dynamic_rnn(
                    cell=cell,
                    dtype=tf.float64,
                    sequence_length=lengths,
                    inputs=inputs_embedded,
                )
            
            outputs_flat = tf.reshape(outputs, [-1, n_cells])
            outputs_flat = tf.nn.dropout(outputs_flat, 0.5)
            
            outputs_flat
            
            fc1_out = tf.contrib.layers.fully_connected(
                                                        outputs_flat,
                                                        n_hidden,
                                                        tf.nn.elu,
                                                        )
                
            fc2_out = tf.contrib.layers.fully_connected(
                                                        fc1_out,
                                                        n_labels,
                                                        None,
                                                        )
                                                        
            logits_flat = fc2_out
            probs_flat = tf.nn.softmax(logits_flat)
            
            # Summaries
            for v in tf.trainable_variables():
                if v.name.startswith(vs.name):
                    tf.histogram_summary(v.name, v)
            
        return logits_flat, probs_flat
        
    def build_loss(self, logits_flat, labels, lengths, partial_classification=True):
        with tf.variable_scope('Loss'):
            if not partial_classification:
                # Mask everthing except the last output for each sequence
                mask = tf.one_hot(lengths-1, tf.to_int32(tf.reduce_max(lengths)), axis=-1, dtype=tf.int64)
            else:
                # Only mask away padding
                mask = tf.sign(labels)
                
            labels_flat = tf.reshape(labels, [-1])
            losses = tf.nn.sparse_softmax_cross_entropy_with_logits(
                        logits_flat, labels_flat)
            mask_flat = tf.reshape(mask, [-1])
            
            masked_losses = tf.mul(tf.to_double(mask_flat), losses)
            masked_losses = tf.reshape(masked_losses, tf.shape(labels))
                
            loss_by_example = tf.reduce_sum(masked_losses, reduction_indices=1)
            if partial_classification:
                loss_by_example = loss_by_example / tf.to_double(lengths)
            mean_loss = tf.reduce_mean(loss_by_example)
            
            loss_summary = tf.scalar_summary('CrossEntropyLoss', mean_loss)
        return mean_loss
            
    def build_accuracy(self, probs_flat, labels, lengths, partial_classification=True):
        with tf.variable_scope('Accuracy'):
            if not partial_classification:
                # Mask everthing except the last output for each sequence
                mask = tf.one_hot(lengths-1, tf.to_int32(tf.reduce_max(lengths)), axis=-1, dtype=tf.int64)
            else:
                # Only mask away padding
                mask = tf.sign(labels)
                
            labels_flat = tf.reshape(labels, [-1])
            correct = tf.equal(tf.argmax(probs_flat, 1), labels_flat)
            mask_flat = tf.reshape(mask, [-1])
            
            masked_correct = tf.to_double(tf.mul(mask_flat, tf.to_int64(correct)))
            masked_correct = tf.reshape(masked_correct, tf.shape(labels))
            
            correct_reduced = tf.reduce_sum(masked_correct, reduction_indices=1) 
            if partial_classification:
                correct_reduced = correct_reduced / tf.to_double(lengths)
            accuracy = tf.reduce_mean(tf.to_double(correct_reduced), name='accuracy')
            
            acc_summary = tf.scalar_summary('Accuracy', accuracy)
        return accuracy
        