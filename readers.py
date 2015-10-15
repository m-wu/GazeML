import csv
import numpy
from abc import ABCMeta, abstractmethod

from experiment import Experiment, Participant, Trial, AOIVisit, Fixation, ValuedPattern, ValuedItem

_author__ = 'Mike Wu'


class EyeTrackingDataReader:
    __metaclass__ = ABCMeta

    def __init__(self):
        self.exp = Experiment()

    @abstractmethod
    def read_all(self):
        """
        Returns an Experiment object that contains the eye-tracking data.
        """
        pass

    def update_user_characteristic_labels(self):
        for char in self.exp.get_user_characteristics():
            char_median = numpy.median([p.get_characteristic_value(char) for p in self.exp.get_all_participants()])
            if char == "expertise":
                char_median = 4
            for participant in self.exp.get_all_participants():
                char_value = participant.get_characteristic_value(char)
                participant.set_characteristic_label(char, 1 if char_value >= char_median else 0)


class BarRadarDataReader(EyeTrackingDataReader):
    def __init__(self):
        EyeTrackingDataReader.__init__(self)
        self.data_folder = "./emdat-data/bar-radar/"

    def read_all(self):
        """
        Returns an Experiment object that contains the eye-tracking data.
        """
        self.read_user_model()
        self.read_emdat_output()
        return self.exp

    def read_user_model(self):
        """
        Read user and task characteristics and store in the Experiment model.
        """
        self.exp.set_user_characteristics(['ps', 'visual', 'verbal', 'expertise'])
        with open(self.data_folder + "NewUMAPTaskDifAnalysis_1.csv", "rb") as profile:
            profile_reader = csv.DictReader(profile)
            for row in profile_reader:
                # Add a new Participant with their characteristics
                self.exp.add_participant(Participant(pid=row["USER_ID"],
                                                     ps=row["ps_score"],
                                                     visual=row["K6"],
                                                     verbal=row["VerbalWM"],
                                                     expertise=row["BarExpert"]))
            self.update_user_characteristic_labels()

    def read_emdat_output(self):
        """
        Read and store the features of each trial in the Experiment model.
        """
        with open(self.data_folder + "emdat_output_br.csv", "rb") as emdat_output:
            emdat_reader = csv.DictReader(emdat_output)
            # Extract gaze feature names and store in the Experiment
            for feature_name in emdat_reader.fieldnames[2:]:  # features start from 3rd column
                if feature_name in ["length", "numsamples", "numsegments", "aoisequence"] or "pathanglesrate" in feature_name:
                    continue  # ignore features

                if "pupil" in feature_name:
                    self.exp.add_feature_name(feature_name, "pupil")
                elif "distance" in feature_name and "path" not in feature_name:
                    self.exp.add_feature_name(feature_name, "head")
                elif "clic" in feature_name or "events" in feature_name or "keypressed" in feature_name or "eyemovementvelocity" in feature_name:
                    pass  # action event features
                else:
                    self.exp.add_feature_name(feature_name, "gaze")

            for row in emdat_reader:
                if row['user'] not in self.exp.get_all_participant_ids():
                    continue
                # Create a new Trial for every task by every user
                trial = Trial(pid=row["user"], tid=row["trial"])

                is_trial_valid = True
                for f_name in self.exp.get_feature_names():
                    if row[f_name] == 'nan':
                        is_trial_valid = False
                    if float(row[f_name]) > 100000:
                            is_trial_valid = False
                    # Add features to the Trial
                    trial.add_feature(f_name, row[f_name])
                # Add AOI scan path as a feature to the Trial
                trial.set_scanpath(row["aoisequence"].replace('low', 'NA'))  # AOI modification

                # Add the Trial to the Experiment if the trial is valid
                if is_trial_valid:
                    self.exp.add_trial(trial)


class InterventionDataReader(EyeTrackingDataReader):
    def __init__(self):
        EyeTrackingDataReader.__init__(self)
        self.data_folder = "./emdat-data/intervention/"

    def read_all(self):
        self.read_user_model()
        self.read_emdat_output()
        # self.read_aoi_visit_data()
        # self.read_fixation_data()
        # self.read_valued_patterns()
        return self.exp

    def read_user_model(self):
        """
        Read user and task characteristics and store in the Experiment model.
        """
        self.exp.set_user_characteristics(['ps', 'visual', 'verbal', 'loc', 'expertise'])
        with open(self.data_folder + "UserCharacteristicScores_InterventionStudy.csv", "rb") as profile:
            profile_reader = csv.DictReader(profile)
            for row in profile_reader:
                # Add a new Participant with their characteristics
                if int(row['ParticipantID']) == 143:
                    continue
                self.exp.add_participant(Participant(pid=row["ParticipantID"],
                                                     ps=row["PS"],
                                                     visual=row["K6"],
                                                     verbal=row["WordSpanInWorkingMemory"],
                                                     locus_of_control=row["LocusOfControl"],
                                                     expertise=row["Expertise_Basic"]))
                self.update_user_characteristic_labels()

    def read_emdat_output(self):
        """
        Read and store the features of each trial in the Experiment model.
        """
        with open(self.data_folder + "emdat_output_None.csv", "rb") as emdat_output:
            emdat_reader = csv.DictReader(emdat_output)
            for feature_name in emdat_reader.fieldnames[6:]:  # features start from 7th column
                if feature_name in ["length", "numsamples", "numsegments"]:
                    continue  # ignore features

                if "pupil" in feature_name:
                    self.exp.add_feature_name(feature_name, "pupil")
                elif "distance" in feature_name and "path" not in feature_name:
                    self.exp.add_feature_name(feature_name, "head")
                elif "clic" in feature_name or "events" in feature_name or "keypressed" in feature_name or "eyemovementvelocity" in feature_name:
                    pass  # action event features
                else:
                    self.exp.add_feature_name(feature_name, "gaze")

            for row in emdat_reader:
                # Create a new Trial for every task by every user
                trial = Trial(pid=row["user"], tid=row["trial"])

                is_trial_valid = True
                for f_name in self.exp.get_feature_names():
                    if row[f_name] == 'nan':
                        is_trial_valid = False
                    # Add features to the Trial
                    trial.add_feature(f_name, row[f_name])
                # Add AOI scan path as a feature to the Trial
                trial.set_scanpath(row["scanpath"])

                # Add the Trial to the Experiment if the trial is valid
                if is_trial_valid:
                    self.exp.add_trial(trial)

    def read_aoi_visit_data(self):
        """
        Read AOI visits data and store in the corresponding trial.
        """
        with open(self.data_folder + "intervention-aoi_visit_data.tsv", "rb") as aoi_visit_data:
            aoi_visit_reader = csv.DictReader(aoi_visit_data, delimiter="\t")
            for row in aoi_visit_reader:
                trial = self.exp.get_trial(row["user"], row["task"])
                if trial is not None:
                    aoi_visit = AOIVisit(abs_time=row["timestamp"],
                                         trial_time=row["trialtime"],
                                         aoi_name=row["aoi"],
                                         duration=row["fixationduration"])
                    trial.add_aoi_visit(aoi_visit)

    def read_fixation_data(self):
        """
        Read fixation data and store in the corresponding trial.
        """
        with open(self.data_folder + "fixation_data-ivt.tsv", "rb") as fixation_data:
            fixation_reader = csv.DictReader(fixation_data, delimiter="\t")
            for row in fixation_reader:
                trial = self.exp.get_trial(row["user"], row["task"])
                if trial is not None:
                    fixation = Fixation(abs_time=row["timestamp"],
                                        trial_time=row["trialtime"],
                                        aoi_name=row["aoi"],
                                        duration=row["fixationduration"])
                    trial.add_fixation(fixation)

    def read_valued_patterns(self):
        with open(self.data_folder + "valued_patterns.tsv", "rb") as valued_pattern_data:
            valued_pattern_reader = csv.reader(valued_pattern_data, delimiter="\t")
            for pattern in valued_pattern_reader:
                items = []
                for item in pattern:
                    aoi, min_duration, max_duration = item.split(',')
                    valued_item = ValuedItem(aoi, min_duration, max_duration)
                    items.append(valued_item)
                pattern = ValuedPattern(list(items))
                self.exp.add_valued_pattern(pattern)
