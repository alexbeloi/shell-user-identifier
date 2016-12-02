import os
import itertools
import random
import tensorflow as tf

def read_bashdata(path='./bash_data/', min_length=2, shuffle=True, 
                  single_out_user = -1, subsequence = False):
    assert os.path.exists(path)    
    def splitter(data):
        split = [item.splitlines() for item in data.split('**EOF**')]
        split = [[i for i in x if i != ''] for x in split]
        return split

    sequences = []
    labels = []
    for i in range(1,10):
        with open(os.path.join(path,'user%d' % (i-1))) as f:
            data = f.read()
            split_data = splitter(data)
            sequences.extend(split_data)
        if single_out_user is -1:
            _labels = [[i]*len(x) for x in split_data]
        else:
            _labels = [[int(i is single_out_user)]*len(x) for x in split_data]
        labels.extend(_labels)
            
    # create dictionary
    dic = {}
    c = 1
    for s in set(itertools.chain.from_iterable(sequences)):
        dic[s] = c
        c += 1

    # map the strings to ints
    sequences_mapped = [map(dic.get, item) for item in sequences]
    
    # compute lengths
    lengths = map(len, sequences_mapped)
    
    #filter by length of session
    sequences_mapped, labels, lengths = zip(*filter(lambda x: x[2] > min_length, 
        zip(sequences_mapped, labels, lengths)))

    # Filter out sequences shorter than min_length
    combined = filter(lambda x: x[2] > min_length, zip(sequences_mapped, labels, lengths))
    if single_out_user is not -1:
        target_user = filter(lambda x: x[1], combined)
        other_users = filter(lambda x: not x[1], combined)
        subsample_other_users = random.sample(other_users, len(target_user))
        combined = target_user + subsample_other_users
    if shuffle:
        random.shuffle(combined)
    inputs, labels, lengths = zip(*combined)
    
    print len(inputs), len(labels), len(lengths)
    
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
    
def create_tfrecords(records_path='.', **kwargs):
    assert os.path.exists(records_path)
    save_path = os.path.join(records_path, 'bash_data.TFRecords')
    
    sequences, label_sequences, _ = read_bashdata(**kwargs)
    with open(save_path, 'w') as fp:
        writer = tf.python_io.TFRecordWriter(fp.name)
    for sequence, label_sequence in zip(sequences, label_sequences):
        ex = make_example(sequence, label_sequence)
        writer.write(ex.SerializeToString())
    writer.close()