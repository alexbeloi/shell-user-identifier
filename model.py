import tensorflow as tf
import numpy as np

N_CLASSES = 2+1
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
            partial_classification=True,
            embeddings=None,
            ):
        
        b_tokens, b_labels, b_lengths = \
            self.batch(inputs, labels, length, batch_size)
        self.batched_labels = b_labels
        self.batched_lengths = b_lengths
        self.logits, self.probs = self.build_model(b_tokens, b_lengths, n_cells, n_hidden, n_stacks, embeddings=embeddings)
        self.loss = self.build_loss(self.logits, b_labels, b_lengths, partial_classification)
        self.acc = self.build_accuracy(self.probs, b_labels, b_lengths, partial_classification)
        
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
        
    def build_model(self, inputs, lengths, n_cells = 64, n_hidden = 32, n_stacks = 1, embeddings = None):
        with tf.variable_scope('Model'):
            if embeddings == None:
                embeddings = tf.get_variable('embeddings', initializer = \
                tf.random_uniform([VOCABULARY_SIZE, EMBEDDING_SIZE],
                    -1.0, 1.0, dtype=tf.float64, seed=1), trainable=True)
            else:
                embeddings = tf.get_variable('embeddings', 
                    initializer = tf.constant(embeddings), trainable=True)
            self.embeddings = embeddings
            tf.histogram_summary("embeddings", embeddings)
                #     -1.0, 1.0, dtype=tf.float64, seed=1))
            # embeddings_x = tf.random_uniform([VOCABULARY_SIZE, EMBEDDING_SIZE],
            #     -1.0, 1.0, dtype=tf.float64, seed=1)
            # embeddings_x = tf.constant(np.identity(VOCABULARY_SIZE))
            
            
            inputs_embedded = tf.nn.embedding_lookup(embeddings, inputs)
            self._inputs_embedded = inputs_embedded
            with tf.variable_scope('lstm') as vs:
                cell = tf.nn.rnn_cell.LSTMCell(num_units=n_cells, state_is_tuple=True)
                cell = tf.nn.rnn_cell.DropoutWrapper(cell=cell, output_keep_prob=0.5)
                cell = tf.nn.rnn_cell.MultiRNNCell(cells=[cell] * n_stacks, state_is_tuple=True)

                outputs, states = tf.nn.dynamic_rnn(
                    cell=cell,
                    dtype=tf.float64,
                    sequence_length=lengths,
                    inputs=inputs_embedded,
                )
                
                # cell summary
                for v in tf.all_variables():
                    if v.name.startswith(vs.name):
                        tf.histogram_summary(v.name, v)
            
            outputs_flat = tf.reshape(outputs, [-1, n_cells])
            outputs_flat = tf.nn.dropout(outputs_flat, 0.5)
            
            out = outputs_flat
            
            W = tf.get_variable('W', initializer = \
                tf.random_normal(
                    [n_cells, n_hidden], stddev=0.01, dtype=tf.float64))
            b = tf.get_variable('b', initializer = \
                tf.random_normal([n_hidden], stddev=0.01, dtype=tf.float64))
            
            
            out = tf.batch_matmul(outputs_flat, W) + b
            
            W2 = tf.get_variable('W2', initializer = \
                tf.random_normal(
                    [n_hidden, N_CLASSES], stddev=0.01, dtype=tf.float64))
            b2 = tf.get_variable('b2', initializer = \
                tf.random_normal([N_CLASSES], stddev=0.01, dtype=tf.float64))
                
            
            # logits_flat = tf.batch_matmul(out, W) + b
            logits_flat = tf.batch_matmul(out, W2) + b2
            probs_flat = tf.nn.softmax(logits_flat)
            
            tf.histogram_summary("w", W)
            tf.histogram_summary("b", b)
            tf.histogram_summary("w2", W2)
            tf.histogram_summary("b2", b2)
            
        return logits_flat, probs_flat
        
    def build_loss(self, logits_flat, labels, lengths, partial_classification=True):
        with tf.variable_scope('Loss'):
            if not partial_classification:
                mask = tf.one_hot(lengths-1, tf.to_int32(tf.reduce_max(lengths)), axis=-1, dtype=tf.int64)
            else:
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
                mask = tf.one_hot(lengths-1, tf.to_int32(tf.reduce_max(lengths)), axis=-1, dtype=tf.int64)
            else:
                mask = tf.sign(labels)
            self.mask = mask
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
        