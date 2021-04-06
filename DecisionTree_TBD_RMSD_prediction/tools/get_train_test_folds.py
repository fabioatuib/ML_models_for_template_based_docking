
import os
import random
# data
import pandas as pd
# math
import numpy as np

def get_train_test_folds(lower_threshold, upper_threshold, number_of_folds = 4, number_of_repeats = 10):
    # get the data set
    df = pd.read_csv('../data/train_test_rmsd_values_featurized_w_sasa_without_bad_pairs.csv')

    # Assign binary labels to the dataset
    df['binned_rmsd'] = None

    for index, rmsd in df[['rmsd']].itertuples():
        if rmsd < 2:
            df.at[index, 'binned_rmsd'] = 1
        elif lower_threshold < rmsd < upper_threshold:
            df.at[index, 'binned_rmsd'] = -1
        else:
            df.at[index, 'binned_rmsd'] = 0

    df = df[df['template']!=df['docked']]

    # Features that will be used to classify:
    not_features = ['template', 'docked', 'rmsd', 'uniprot_id', 'mcs_smartsString',
                    'smiles_template', 'smiles_docked', 'binned_rmsd']
    features = df.drop(columns=not_features).columns.tolist()
    print(features)
    # TODO add option to remove rows that have "binned_rmsd" = -1
    no_inconcl_df = df.loc[df['binned_rmsd']!=-1].reset_index(drop=True)
    #no_inconcl_df = df.reset_index(drop=True)

    groups = []
    size_of_groups = {}
    for group in no_inconcl_df['group'].drop_duplicates():
        groups +=  [group]
        size_of_groups[group] = no_inconcl_df[no_inconcl_df['group']==group].shape[0]

    print(size_of_groups)

    def check_sizes_of_folds(folds):
        sizes = []
        for fold in folds:
            sizes += [sum([size_of_groups[g] for g in fold])]
        return sizes

    k = number_of_folds

    base_size = len(groups)//(k)
    base_rest = len(groups)%(k)

    print(base_size, base_rest)

    train_test_folds_dfs = []
    train_test_folds_indexes = []
    for number in range(number_of_repeats):

        random.shuffle(groups)
        folds = [groups[i*base_size:(i+1)*base_size] for i in range(k)]

        for i in range(base_rest):
            folds[-(i+1)] = np.append(groups[-(i+1)], folds[-(i+1)])

        print(check_sizes_of_folds(folds))

        folds_dfs = [[]]
        for fold in folds:
            for group in fold:
                folds_dfs[-1] += [no_inconcl_df.loc[no_inconcl_df['group']==group]]
            folds_dfs[-1] = pd.concat(folds_dfs[-1])
            folds_dfs += [[]]

        _train_test_folds_dfs = []
        for i in range(k):
            _train_test_folds_dfs += [[]]
            _train_test_folds_dfs[-1] += [pd.concat([folds_dfs[j] for j in range(k) if j!=i])]
            _train_test_folds_dfs[-1] += [folds_dfs[i]]

        _train_test_folds_indexes = []
        for train, test in _train_test_folds_dfs:
            _train_test_folds_indexes += [[train.index.values, test.index.values]]

        train_test_folds_dfs += _train_test_folds_dfs
        train_test_folds_indexes += _train_test_folds_indexes

    return no_inconcl_df, train_test_folds_indexes, train_test_folds_dfs