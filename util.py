import os
import itertools
import random
import tensorflow as tf
from sklearn.model_selection import train_test_split

def read_bashdata(path='./bash_data/', min_length=2, shuffle=True, 
                  single_out_user = -1):
    sequences = []
    labels = []
    for i in range(1,10):
        with open(os.path.join(path,'user%d' % (i-1))) as f:
            data = f.read()
            split_data = [item.splitlines() for item in data.split('**EOF**')]
            split_data = \
                [[j for j in x if i != ''] + ['**EOF**'] for x in split_data]
            sequences.extend(split_data)
        if single_out_user == -1:
            _labels = [[i]*len(x) for x in split_data]
        else:
            _labels = [[int(i == single_out_user)+1]*len(x) for x in split_data]
        labels.extend(_labels)
            
    # create dictionary
    dic = {}
    index = 1
    for s in itertools.chain.from_iterable(sequences):
        if s not in dic:
            dic[s] = index
            index += 1

    # map the strings to ints
    sequences_mapped = [map(dic.get, item) for item in sequences]
    
    # compute lengths
    lengths = map(len, sequences_mapped)
    
    #filter by length of session
    sequences_mapped, labels, lengths = zip(*filter(lambda x: x[2] > min_length, 
        zip(sequences_mapped, labels, lengths)))

    # Filter out sequences shorter than min_length
    combined = zip(sequences_mapped, labels, lengths)
    if single_out_user is not -1:
        target_user = filter(lambda x: x[1][0]==2, combined)
        other_users = filter(lambda x: x[1][0]!=2, combined)
        subsample_other_users = random.sample(other_users, len(target_user))
        combined = target_user + subsample_other_users
    if shuffle:
        random.shuffle(combined)
    inputs, labels, lengths = zip(*combined)
    
    return inputs, labels, lengths
    
def make_example(sequence, labels):
    # The object we return
    ex = tf.train.SequenceExample()
    # A non-sequential feature of our example
    sequence_length = len(sequence)
    ex.context.feature["length"].int64_list.value.append(sequence_length)
    # Feature lists for the two sequential features of our example
    fl_tokens = ex.feature_lists.feature_list["tokens"]
    fl_labels = ex.feature_lists.feature_list["labels"]
    for token, label in zip(sequence, labels):
        fl_tokens.feature.add().int64_list.value.append(token)
        fl_labels.feature.add().int64_list.value.append(label)
    return ex
    
def parse_example(ex):
    context_features = {
        "length": tf.FixedLenFeature([], dtype=tf.int64)
    }
    sequence_features = {
        "tokens": tf.FixedLenSequenceFeature([], dtype=tf.int64),
        "labels": tf.FixedLenSequenceFeature([], dtype=tf.int64)
    }

    # Parse the example (returns a dictionary of tensors)
    context_parsed, sequence_parsed = tf.parse_single_sequence_example(
        serialized=ex,
        context_features=context_features,
        sequence_features=sequence_features
    )
    return sequence_parsed, context_parsed
    
def create_tfrecords(save_path='.', 
                     prefix='bash_data', 
                     test_size=0.2, 
                     **kwargs):
    assert os.path.exists(save_path)
    sequences, label_sequences, _ = read_bashdata(**kwargs)
    x_train, x_test, y_train, y_test = \
        train_test_split(sequences, 
                         label_sequences, 
                         test_size=test_size, 
                         random_state=1)
    
    def write_examples(x, y, filename):
        file_path = os.path.join(save_path, filename)
        with open(file_path, 'w') as fp:
            writer = tf.python_io.TFRecordWriter(fp.name)
        for sequence, label_sequence in zip(x, y):
            ex = make_example(sequence, label_sequence)
            writer.write(ex.SerializeToString())
        writer.close()
        
    write_examples(x_train, y_train, prefix + '_train.TFRecords')
    write_examples(x_test, y_test, prefix + '_test.TFRecords')
    
    