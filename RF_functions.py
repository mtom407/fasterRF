"""This module contatins vectorized/faster versions of functions presented in:
https://machinelearningmastery.com/implement-random-forest-scratch-python/"""

import pandas as pd
import numpy as np

def load_data(filename):
    """Load data into lists."""
    targets = []
    strings = []

    with open(filename) as f:
        lines = f.readlines()
        for line in lines[1:]:
            if len(line) == 2:
                targets.append(line[:-1])
            else:
                strings.append(line[:-1])
    return strings, targets

def create_seq_df(sequences, targets, numerical=False):
    """Transofrm list data into a pandas DataFrame."""
    dna_seq_len = len(sequences[0])
    columns = []
    for i in range(dna_seq_len):
        columns.append(f'pos{i}')

    columns.append('target')

    if numerical:
        all_sequences = []
        for (string, target) in zip(sequences, targets):
            dna_seq = [ord(x) for x in string] + list(target)
            all_sequences.append(dna_seq)
        data_frame = pd.DataFrame(data=all_sequences, columns=columns)
    else:
        all_sequences = []
        for (string, target) in zip(sequences, targets):
            dna_seq = list(string) + list(target)
            all_sequences.append(dna_seq)
        data_frame = pd.DataFrame(data=all_sequences, columns=columns)

    return data_frame

def encode_targets(dataset):
    """Change string target characters into integers."""
    dataset['target'] = dataset['target'].apply(int)
    return dataset


def train_test_split(dataset, split_fraction):
    """Shuffle and split data into train and test sets."""
    # miejsce podziału
    split_point = int(len(dataset)*split_fraction)
    # przemieszanie przykładów
    dataset_shuffle = dataset.sample(frac=1).reset_index(drop=True)
    # stworzenie zbioru testowego i uczącego
    train_set = dataset_shuffle[:split_point]
    test_set = dataset_shuffle[split_point:]

    return train_set, test_set


def accuracy_metric(actual, predicted):
    """Calculate mean of good predictions as the accuracy metric."""
    return np.mean(actual == predicted)*100


def test_split(index, value, dataset):

    # macierz logiczna
    dataset_log = dataset < value

    # indeksy przykładów do lewej i prawej podgrupy
    to_left = dataset_log[:, index]
    to_right = np.logical_not(to_left)
    # przykłady do lewej i prawej podgrupy
    left = dataset[np.asarray(to_left)]
    right = dataset[np.asarray(to_right)]

    return left, right

def gini_index(groups):#, classes):
    """Calculate gini index for every group passed as input."""
    # liczba probek do podzielenia
    n_instances = float(len(groups[0]) + len(groups[1]))

    gini = 0.0
    for group in groups:
        size = float(len(group))
        if size == 0:
            continue
        _, counts = np.unique(group[:, -1], return_counts=True)
        counts = counts / size
        score = np.dot(counts, counts)

        gini += (1.0 - score) * (size / n_instances)
    return gini

def find_root_variable(dataset, n_features):
    # wszystkie wartości etykiet
    #class_values = np.unique(dataset[:][-1])
    b_index, b_value, b_score, b_groups = 999, 999, 999, None
    # lista wybranych cech
    features = list(np.random.choice(range(len(dataset[0])-1), n_features, replace=False))

    # znalezienie najlepszej cechy
    for index in features:
        for row_id in range(len(dataset)):
            # podział na podgrupy
            groups = test_split(index, dataset[row_id][index], dataset)
            # obliczenie współczynnika
            gini = gini_index(groups)#, class_values)
            # zapisanie najlepszego wyniku
            if gini < b_score:
                b_index, b_value, b_score, b_groups = index, dataset[row_id][index], gini, groups
    return {'index':b_index, 'value':b_value, 'groups':b_groups}



def to_terminal(group):
    """Convert node into terminal node"""
    targets, counts = np.unique(group[:, -1], return_counts=True)
    return targets[np.argmax(counts)]


def split(node, max_depth, min_size, n_features, depth):
    left, right = node['groups']
    del node['groups']

    if left.shape[0] == 0 or right.shape[0] == 0:
        node['left'] = node['right'] = to_terminal(np.vstack((left, right)))
        return

    if depth >= max_depth:
        node['left'], node['right'] = to_terminal(left), to_terminal(right)
        return

    if len(left) <= min_size:
        node['left'] = to_terminal(left)
    else:
        node['left'] = find_root_variable(left, n_features)
        split(node['left'], max_depth, min_size, n_features, depth+1)

    if len(right) <= min_size:
        node['right'] = to_terminal(right)
    else:
        node['right'] = find_root_variable(right, n_features)
        split(node['right'], max_depth, min_size, n_features, depth+1)


def build_tree(train, max_depth, min_size, n_features):
    root = find_root_variable(train, n_features)
    split(root, max_depth, min_size, n_features, 1)
    return root

def predict(node, row):
    if row[node['index']] < node['value']:
        if isinstance(node['left'], dict):
            return predict(node['left'], row)
        else:
            return node['left']
    else:
        if isinstance(node['right'], dict):
            return predict(node['right'], row)
        else:
            return node['right']

def bagging_predict(trees, row):
    predictions = [predict(tree, row) for tree in trees]
    preds, count = np.unique(predictions, return_counts=True)
    return preds[np.argmax(count)]

def random_forest(train, test, max_depth, min_size, sample_size, n_trees, n_features):
    trees = list()
    for _ in range(n_trees):

        sample_vec = np.random.choice(np.arange(len(train)),
                                      np.int(sample_size*train.shape[0]),
                                      replace=True)
        sample = train[sample_vec]

        tree = build_tree(sample, max_depth, min_size, n_features)

        trees.append(tree)

    predictions = np.array([bagging_predict(trees, row) for row in test])
    return predictions


def run_rf(dataset, n_trees, max_depth, min_size, sample_size):
    dataset = encode_targets(dataset)

    n_features = np.int(np.ceil(np.sqrt(dataset.shape[1]-1)))

    train_set, test_set = train_test_split(dataset, 0.7)
    train_set = train_set.to_numpy()
    test_set = test_set.to_numpy()

    predicted = random_forest(train_set, test_set,
                              max_depth, min_size, sample_size, n_trees, n_features)
    actual = test_set[:, -1]

    accuracy = accuracy_metric(actual, predicted)
    print('Trees: {}, Max depth: {}, Min size: {}, Sample size: {}'.format(n_trees,
                                                                           max_depth,
                                                                           min_size,
                                                                           sample_size))
    print('Score: %s' % accuracy)
    print()
    return n_trees, accuracy
