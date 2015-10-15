import csv
from collections import defaultdict
import itertools

import numpy
from sklearn import cross_validation
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, recall_score
from sklearn.svm import SVC

from constants import *
import features
from readers import BarRadarDataReader, InterventionDataReader

_author__ = 'Mike Wu'


DATASETS = [BAR_RADAR, INTERVENTION]
CLASS_NAMES = [PS, VERBAL, VISUAL, LOC, EXPERTISE]

CLASSIFIERS = [MAJORITY_CLASS, LOGISTIC_REGRESSION, RANDOM_FOREST]
SCORE_TYPES = [ACCURACY, RECALL_C0, RECALL_C1]

# Feature set groups
SUMMATIVE_FEATURE_SET_CONFIGS = [(GAZE,), (PUPIL,), (HEAD,), (PUPIL, HEAD), (GAZE, HEAD), (GAZE, PUPIL, HEAD)]
SEQUENCE_FEATURE_SET_CONFIGS = [(SEQUENCE,), (GAZE,), (GAZE, SEQUENCE), (PUPIL, HEAD), (PUPIL, HEAD, SEQUENCE)]

# Sequence pattern selection criteria
S_SUPPORTS = [0.4, 0]  # minimum S-Support threshold
ALPHAS = [0.05, 1]  # alpha level for the statistical tests on s-support and average pattern frequency comparisons

SPLIT_BY = SPLIT_BY_USER  # either SPLIT_BY_TRIAL or SPLIT_BY_USER

OUTPUT_FOLDER = "../GazeML_output/"
OUTPUT_FILE_NAME = "accuracy-summative-pct-{folds}cv-{reps}rep"
OUTPUT_FILE_EXT = ".tsv"
OUTPUT_WRITER = None
OUTPUT_HEADER = ["dataset", "char", "feature", "ss", "alpha", "classifier", "accuracy", "class_0", "class_1"]

PRINT_STATUS = True  # whether to print status to console
PRINT_OUTPUT = True  # whether to print output to console
WRITE_OUTPUT = False  # whether to write output to file

CV_FOLDS = 10  # number of folds in a k-fold cross-validation
CV_REPS = 1  # number of runs (repetitions) of k-fold cross-validation


def main():
    """
    Set up experimental conditions, open output file, read in data, generate sequence pattern features, run cross-
     validation and output results.
    :return: None
    """
    # Select a subset of the experimental conditions if needed.
    datasets = DATASETS
    class_names = CLASS_NAMES[0:3]
    feature_set_configs = SUMMATIVE_FEATURE_SET_CONFIGS
    s_support = S_SUPPORTS
    alphas = ALPHAS
    exp_params = generate_experimental_conditions(datasets, class_names, feature_set_configs, s_support, alphas)

    print_status("Classes: {classes}".format(classes=class_names))
    print_status("Features: {feature_set_configs}".format(feature_set_configs=feature_set_configs))

    if WRITE_OUTPUT:
        # Set up output file
        output_file_name = OUTPUT_FILE_NAME.format(folds=CV_FOLDS, reps=CV_REPS)
        output_file_path = OUTPUT_FOLDER + output_file_name + OUTPUT_FILE_EXT
        f = open(output_file_path, 'a')
        global OUTPUT_WRITER
        OUTPUT_WRITER = csv.writer(f, delimiter='\t')
        write_output(OUTPUT_HEADER)
        print_status("Output file path: {path}".format(path=output_file_path))
    else:
        f = None

    print_status("Start reading files")
    experiments = {}  # stores the eye-tracking datasets in the form of Experiment models
    if BAR_RADAR in DATASETS:
        experiments[BAR_RADAR] = BarRadarDataReader().read_all()
    if INTERVENTION in DATASETS:
        experiments[INTERVENTION] = InterventionDataReader().read_all()
    print_status("Finish reading files")

    # Calculate and add additional features for each dataset
    for experiment in experiments.values():
        features.populate_sequence_pattern_frequency(experiment)  # if using sequence pattern features
        # features.add_valued_pattern_features(experiments)
        # features.add_long_aoi_visit_features(experiments)

    for exp_param in exp_params:
        perform_cross_validation(experiments[exp_param[DATASET]], exp_param)

    if f:
        f.close()


def generate_experimental_conditions(datasets, class_names, feature_set_configs, s_support, alpha):
    """
    Generate parameter sets that the program can iterate over.
    :param datasets: list of names of the datasets
    :param class_names: list of names of the target classes
    :param feature_set_configs: list of feature set configurations
    :param s_support: list of s_support values
    :param alpha: list of critical p-values
    :return: a list of dictionaries each contains one experimental condition
    """
    factor_names = [DATASET, CLASS_NAME, FEATURE_SET, S_SUPPORT, ALPHA]
    conditions = list(itertools.product(*[datasets, class_names, feature_set_configs, s_support, alpha]))
    conditions_dict = [dict(zip(factor_names, condition)) for condition in conditions]  # convert list to dictionary
    return conditions_dict


def perform_cross_validation(exp, exp_conditions):
    """
    Perform cross-validations and output results.
    :param exp: an Experiment object that contains the eye-tracking data
    :param exp_conditions: a dictionary that contains the experimental conditions
    :return: None
    """
    class_name = exp_conditions[CLASS_NAME]
    feature_set = exp_conditions[FEATURE_SET]
    ss = exp_conditions[S_SUPPORT]
    alpha = exp_conditions[ALPHA]

    x = create_feature_set(exp, exp.get_feature_names(feature_set))  # get all features except sequence features
    y = create_target_list(exp, class_name)
    data = numpy.array(x)
    target = numpy.array(y)

    scores = defaultdict(lambda: defaultdict(list))

    for i in range(CV_REPS):
        fold_scores = defaultdict(list)

        for train_index, test_index in get_train_test_index(target=target, exp=exp):
            if SEQUENCE in feature_set:
                # Perform pattern selection and add pattern as features:
                sequence_x = features.get_sequence_pattern_features(exp, class_name, train_index, ss, alpha)
                x_with_sequence = []
                for j in range(len(x)):
                    x_with_sequence.append(x[j] + sequence_x[j])
                data = numpy.array(x_with_sequence)

            for classifier in CLASSIFIERS:
                fold_scores[classifier].append(
                    get_scores(get_classifier(classifier), data, target, train_index, test_index))

        # calculate the means of the cross-validation
        for classifier in CLASSIFIERS:
            for score_type in SCORE_TYPES:
                mean_score = numpy.average([fold[score_type] for fold in fold_scores[classifier]])
                scores[score_type][classifier].append(mean_score)

    output_cv_scores(exp_conditions, scores)


def get_train_test_index(target=None, exp=None):
    """
    Generate training and test trial indices, based on splitting by trial or by user.
    :param target: a list of classification target values, based on which returns stratified K-Folds for split by user.
    :param exp: the experiment object used to split trials by user.
    :return: a generator that produces training and testing trial indices.
    """
    if SPLIT_BY_TRIAL:
        for train_index, test_index in cross_validation.StratifiedKFold(target, n_folds=CV_FOLDS, shuffle=True):
            yield train_index, test_index
    if SPLIT_BY_USER:
        all_pids = numpy.array(exp.get_all_participant_ids())
        for train_index, test_index in cross_validation.KFold(len(all_pids), n_folds=CV_FOLDS, shuffle=True):
            train_pid = all_pids[train_index]
            test_pid = all_pids[test_index]
            train_trial_id = get_trial_ids_for_users(exp, train_pid)
            test_trial_id = get_trial_ids_for_users(exp, test_pid)
            yield train_trial_id, test_trial_id


def get_trial_ids_for_users(exp, pids):
    """
    Get trial indices for a set of users.
    :param exp: the experiment object
    :param pids: a list of user ids
    :return: a list of ids of the trials belonging to the list of users in pids.
    """
    trial_ids = []
    i = 0
    for trial in exp.get_all_trials():
        if trial.get_pid() in pids:
            trial_ids.append(i)
        i += 1
    return trial_ids


def create_feature_set(exp, feature_names):
    """
    Convert features from all trials into a 2D list.
    """
    x = []
    for trial in exp.get_all_trials():
        sample = []
        for feature in feature_names:
            feature_value = trial.get_feature(feature)
            sample.append(feature_value)
        x.append(sample)
    return x


def create_target_list(exp, class_name):
    """
    Convert prediction target into a list.
    """
    y = []
    for trial in exp.get_all_trials():
        participant = exp.get_participant(trial.get_pid())
        if class_name in exp.get_user_characteristics():
            y.append(participant.get_characteristic_label(class_name))
    return y


def get_classifier(classifier_name):
    """
    Return the classifier based on the name of the classifier.
    :param classifier_name: name of the classifier
    :return: a classifier object.
    """
    if classifier_name == RANDOM_FOREST:
        return RandomForestClassifier(n_estimators=100, oob_score=True)
    elif classifier_name == LOGISTIC_REGRESSION:
        return LogisticRegression()
    elif classifier_name == MAJORITY_CLASS:
        return DummyClassifier(strategy="most_frequent")
    elif classifier_name == SVM:
        return SVC()
    elif classifier_name == GTB:
        return GradientBoostingClassifier()
    elif classifier_name == EXTRA_TREES:
        return ExtraTreesClassifier()
    return None


def get_scores(classifier, data, target, train_index, test_index):
    """
    Train and test classifiers. Return performance scores.
    :param classifier: classifier object
    :param data: feature array
    :param target: label array
    :param train_index: training set indices
    :param test_index: testing set indices
    :return: a dictionary containing various scores
    """
    scores = {}
    # train classifier
    classifier.fit(data[train_index], target[train_index])
    # predict on test set
    prediction = classifier.predict(data[test_index])
    # compute scores
    if ACCURACY in SCORE_TYPES:
        scores[ACCURACY] = accuracy_score(target[test_index], prediction)
    if RECALL_C0 in SCORE_TYPES:  # class accuracy
        scores[RECALL_C0] = recall_score(target[test_index], prediction, pos_label=0)
    if RECALL_C1 in SCORE_TYPES:  # class accuracy
        scores[RECALL_C1] = recall_score(target[test_index], prediction, pos_label=1)
    return scores


def output_cv_scores(exp_param, scores):
    """
    Print or output cross-validation scores to console or to file.
    :param exp_param: experimental parameters
    :param scores: a dictionary of scores from the get_scores() function
    :return: None
    """
    for i in range(CV_REPS):
        dataset = exp_param[DATASET]
        class_name = exp_param[CLASS_NAME]
        feature_set = exp_param[FEATURE_SET]

        parameters = [dataset, class_name, feature_set, exp_param[S_SUPPORT], exp_param[ALPHA]]

        for classifier in CLASSIFIERS:
            output_line = parameters + [classifier, scores[ACCURACY][classifier][i], scores[RECALL_C0][classifier][i],
                                        scores[RECALL_C1][classifier][i]]
            print_output(output_line)
            write_output(output_line)


def print_status(message):
    if PRINT_STATUS:
        print message


def print_output(line):
    if PRINT_OUTPUT:
        print line


def write_output(line):
    if WRITE_OUTPUT:
        if not len(line) == len(OUTPUT_HEADER):
            print "Output line does not match the header."
        OUTPUT_WRITER.writerow(line)


if __name__ == '__main__':
    main()
