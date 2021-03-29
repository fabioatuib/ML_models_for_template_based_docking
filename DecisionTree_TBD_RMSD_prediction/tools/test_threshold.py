import random
# math
import numpy as np
# Machine learning
import sklearn
from sklearn import tree
# plotting
from matplotlib import pyplot as plt


def test_threshold(train_test, maximum_depth, evaluation_thresholds, thresholds, binned_rmsd, not_features,
                   show_graphs=False):

    clf = tree.DecisionTreeClassifier(max_depth=maximum_depth, class_weight="balanced")

    precision_with_thresh = {}
    recall_with_thresh = {}
    f1_score_with_thresh = {}
    roc_auc_score_with_thresh = {}
    precision_mean_with_thresh = {}
    recall_mean_with_thresh = {}
    f1_score_mean_with_thresh = {}
    roc_auc_score_mean_with_thresh = {}
    precision_std_with_thresh = {}
    recall_std_with_thresh = {}
    f1_score_std_with_thresh = {}
    roc_auc_score_std_with_thresh = {}
    for evaluation_threshold in evaluation_thresholds:
        precision_with_thresh[evaluation_threshold] = {thresh:[] for thresh in thresholds}
        recall_with_thresh[evaluation_threshold] = {thresh:[] for thresh in thresholds}
        f1_score_with_thresh[evaluation_threshold] = {thresh:[] for thresh in thresholds}
        roc_auc_score_with_thresh[evaluation_threshold] = {thresh:[] for thresh in thresholds}

        precision_mean_with_thresh[evaluation_threshold] = {thresh:[] for thresh in thresholds}
        recall_mean_with_thresh[evaluation_threshold] = {thresh:[] for thresh in thresholds}
        f1_score_mean_with_thresh[evaluation_threshold] = {thresh:[] for thresh in thresholds}
        roc_auc_score_mean_with_thresh[evaluation_threshold] = {thresh:[] for thresh in thresholds}

        precision_std_with_thresh[evaluation_threshold] = {thresh:[] for thresh in thresholds}
        recall_std_with_thresh[evaluation_threshold] = {thresh:[] for thresh in thresholds}
        f1_score_std_with_thresh[evaluation_threshold] = {thresh:[] for thresh in thresholds}
        roc_auc_score_std_with_thresh[evaluation_threshold] = {thresh:[] for thresh in thresholds}

    for threshold in thresholds:
        for train, test in train_test:
            _feats_train = train.drop(columns=not_features + ['group']).values.tolist()
            _labels_train = train[['binned_rmsd']]
            for index in _labels_train.index:
                _labels_train.at[index, 'binned_rmsd'] = binned_rmsd[threshold][index]
            _labels_train = _labels_train.values.tolist()
            clf = clf.fit(_feats_train, _labels_train)
            _feats_test = test.drop(columns=not_features + ['group']).values.tolist()
            for evaluation_threshold in evaluation_thresholds:
                _labels_test = test[['binned_rmsd']]
                for index in _labels_test.index:
                    _labels_test.at[index, 'binned_rmsd'] = binned_rmsd[evaluation_threshold][index]
                _labels_test = _labels_test.values.tolist()
                clf_predict = clf.predict(_feats_test)

                precision_with_thresh[evaluation_threshold][threshold] +=\
                    [sklearn.metrics.precision_score(_labels_test, clf_predict)]

                recall_with_thresh[evaluation_threshold][threshold] +=\
                    [sklearn.metrics.recall_score(_labels_test, clf_predict)]

                f1_score_with_thresh[evaluation_threshold][threshold] +=\
                    [sklearn.metrics.f1_score(_labels_test, clf_predict)]

                roc_auc_score_with_thresh[evaluation_threshold][threshold] +=\
                    [sklearn.metrics.roc_auc_score(_labels_test, clf.predict_proba(_feats_test)[:, 1])]

        for evaluation_threshold in evaluation_thresholds:
            precision_mean_with_thresh[evaluation_threshold][threshold] =\
                np.mean(precision_with_thresh[evaluation_threshold][threshold])
            recall_mean_with_thresh[evaluation_threshold][threshold] =\
                np.mean(recall_with_thresh[evaluation_threshold][threshold])
            f1_score_mean_with_thresh[evaluation_threshold][threshold] =\
                np.mean(f1_score_with_thresh[evaluation_threshold][threshold])
            roc_auc_score_mean_with_thresh[evaluation_threshold][threshold] =\
                np.mean(roc_auc_score_with_thresh[evaluation_threshold][threshold])
            precision_std_with_thresh[evaluation_threshold][threshold] =\
                np.std(precision_with_thresh[evaluation_threshold][threshold])
            recall_std_with_thresh[evaluation_threshold][threshold] =\
                np.std(recall_with_thresh[evaluation_threshold][threshold])
            f1_score_std_with_thresh[evaluation_threshold][threshold] =\
                np.std(f1_score_with_thresh[evaluation_threshold][threshold])
            roc_auc_score_std_with_thresh[evaluation_threshold][threshold] =\
                np.std(roc_auc_score_with_thresh[evaluation_threshold][threshold])
            #print('Threshold', threshold, 'done!', recall_mean_with_thresh[threshold], precision_mean_with_thresh[threshold], roc_auc_score_mean_with_thresh[threshold])

    if show_graphs==True:
        for evaluation_threshold in evaluation_thresholds:

            plt.errorbar(thresholds, [recall_mean_with_thresh[evaluation_threshold][threshold] for threshold in thresholds],
                         yerr=[recall_std_with_thresh[evaluation_threshold][threshold] for threshold in thresholds],
                         capsize=5, fmt='--o', label='recall')

            plt.errorbar(thresholds, [precision_mean_with_thresh[evaluation_threshold][threshold] for threshold in thresholds],
                         yerr=[precision_std_with_thresh[evaluation_threshold][threshold] for threshold in thresholds],
                         capsize=5, fmt='--o', label='precision')

            plt.plot(thresholds, [f1_score_mean_with_thresh[evaluation_threshold][threshold] for threshold in thresholds],
                     label='f1_score')

            plt.legend()
            plt.xlabel('threshold')
            plt.ylabel('score')
            plt.title('performance on classifying below '+str(evaluation_threshold)+' rmsd with a DT of maximum depth '+str(maximum_depth))
            plt.show()

            plt.errorbar(thresholds, [roc_auc_score_mean_with_thresh[evaluation_threshold][threshold] for threshold in thresholds],
                         yerr=[roc_auc_score_std_with_thresh[evaluation_threshold][threshold] for threshold in thresholds],
                         capsize=5, fmt='--o', label='roc_auc_score')
            plt.legend()
            plt.xlabel('threshold')
            plt.ylabel('score')
            plt.title('performance on classifying below '+str(evaluation_threshold)+' rmsd with a DT of maximum depth '+str(maximum_depth))
            plt.show()

    return {'precision': precision_with_thresh,
            'recall': recall_with_thresh,
            'f1_score': f1_score_with_thresh,
            'roc_auc_score': roc_auc_score_with_thresh,
            'precision_mean': precision_mean_with_thresh,
            'recall_mean': recall_mean_with_thresh,
            'f1_score_mean': f1_score_mean_with_thresh,
            'roc_auc_score_mean': roc_auc_score_mean_with_thresh,
            'precision_std': precision_std_with_thresh,
            'recall_std': recall_std_with_thresh,
            'f1_score_std': f1_score_std_with_thresh,
            'roc_auc_score_std': roc_auc_score_std_with_thresh}
