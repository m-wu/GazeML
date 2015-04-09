import csv
import numpy
from sklearn import cross_validation
from sklearn.ensemble import RandomForestClassifier
from sklearn.dummy import DummyClassifier
from experiment import Experiment, Participant, Task, Trial, AOIVisit

data_folder = "./emdat-data/intervention-ivt-v1/"


def main():
    exp = Experiment()

    read_user_task_model(exp)
    read_emdat_output(exp)
    read_aoi_visit_data(exp)

    # add_long_aoi_visit_features(exp)

    x = create_feature_set(exp, exp.get_feature_names())
    y = create_target_list(exp)
    accuracy, baseline = perform_cv(x, y)
    print "Accuracy: {accuracy}".format(accuracy=accuracy)
    print "Baseline: {baseline}".format(baseline=baseline)


def read_user_task_model(exp):
    """
    Read user and task characteristics and store in the Experiment model.
    """
    with open(data_folder + "user_task_profile.csv", "rb") as profile:
        profile_reader = csv.DictReader(profile)
        for row in profile_reader:
            # Add a new Participant with their characteristics
            exp.add_participant(Participant(pid=row["USER_ID"],
                                            ps=row["ps_score"],
                                            visual=row["K6"],
                                            verbal=row["VerbalWM"]))
            # Add a new Task with its characteristics
            exp.add_task(Task(tid=row["TaskName"], difficulty=row["task_diff"]))


def read_emdat_output(exp):
    """
    Read and store the features of each trial in the Experiment model.
    """
    with open(data_folder + "emdat_output_None.csv", "rb") as emdat_output:
        emdat_reader = csv.DictReader(emdat_output)
        # Extract gaze feature names and store in the Experiment
        exp.add_feature_names(emdat_reader.fieldnames[7:])

        for row in emdat_reader:
            # Create a new Trial for every task by every user
            trial = Trial(pid=row["user"], tid=row["trial"])

            is_trial_valid = True
            for f_name in exp.get_feature_names():
                if row[f_name] == 'nan':
                    is_trial_valid = False
                # Add features to the Trial
                trial.add_feature(f_name, row[f_name])
            # Add AOI scanpath as a feature to the Trial
            trial.add_scanpath(row["scanpath"])

            # Add the Trial to the Experiment if the trial is valid
            if is_trial_valid:
                exp.add_trial(trial)


def read_aoi_visit_data(exp):
    """
    Read AOI visits data and store in the corresponding trial.
    """
    with open(data_folder + "intervention-aoivisitdata.tsv", "rb") as aoivisitdata:
        aoivisit_reader = csv.DictReader(aoivisitdata, delimiter="\t")
        for row in aoivisit_reader:
            trial = exp.get_trial(row["user"], row["task"])
            if trial is not None:
                aoivisit = AOIVisit(abs_time=row["timestamp"],
                                    trial_time=row["trialtime"],
                                    aoi_name=row["aoi"],
                                    duration=row["fixationduration"])
                trial.add_aoivisit(aoivisit)


def create_feature_set(exp, features):
    """
    Convert features from all trials into a 2D list.
    """
    x = []
    for trial in exp.get_all_trials():
        sample = []
        for feature in features:
            feature_value = trial.get_feature(feature)
            sample.append(feature_value)
        x.append(sample)
    return x


def create_target_list(exp):
    """
    Convert prediction target into a list.
    """
    y = []
    for trial in exp.get_all_trials():
        ps = exp.get_participant(trial.get_pid()).get_visual()
        y.append(ps)
    return split_median(y)


def split_median(l):
    """
    Convert a list of numbers into a list of 0's and 1's, 
    1 if the element is above the median, and 0 otherwise.
    """
    median = numpy.median(l)
    return [1 if item > median else 0 for item in l]


def perform_cv(x, y):
    """
    Perform K-fold cross validation.
    """
    K = 10
    n = len(x)

    data = numpy.array(x)
    target = numpy.array(y)

    scores = []
    baselines = []

    for train_index, test_index in cross_validation.KFold(n, n_folds=K, shuffle=True):
        # Find the baseline with majority-class classifier
        bl_clf = DummyClassifier(strategy='most_frequent')
        bl_clf.fit(data[train_index], target[train_index])
        baselines.append(bl_clf.score(data[test_index], target[test_index]))

        # Perform classification and stores the accuracy
        clf = RandomForestClassifier()
        clf.fit(data[train_index], target[train_index])
        scores.append(clf.score(data[test_index], target[test_index]))

    return numpy.average(scores), numpy.average(baselines)


if __name__ == '__main__':
    main()