_author__ = 'Mike Wu'


class Experiment(object):
    def __init__(self):
        self.trials = {}
        self.participants = {}
        self.tasks = {}
        self.feature_names = {}
        self.valued_patterns = []
        self.user_characteristics = []

    def add_participant(self, participant):
        p_id = participant.get_id()
        if p_id not in self.participants:
            self.participants[participant.get_id()] = participant

    def get_participant(self, pid):
        return self.participants.get(pid, None)

    def get_all_participant_ids(self):
        return self.participants.keys()

    def get_all_participants(self):
        return self.participants.values()

    def add_task(self, task):
        task_id = task.get_id()
        if task_id not in self.tasks:
            self.tasks[task.get_id()] = task

    def add_feature_name(self, feature_name, feature_set):
        self.feature_names.setdefault(feature_set, []).append(feature_name)

    def add_feature_names(self, feature_names, feature_set):
        self.feature_names.setdefault(feature_set, []).extend(feature_names)

    def get_feature_names(self, feature_sets=None):
        if feature_sets is None:
            return [name for f_set in self.feature_names.values() for name in f_set]

        return [name for f_set in feature_sets for name in self.feature_names.get(f_set, [])]

    def add_trial(self, trial):
        trial_id = trial.get_trial_id()
        if trial_id not in self.trials:
            self.trials[trial_id] = trial

    def get_trial(self, pid, tid):
        trial_id = (pid, tid)
        return self.trials.get(trial_id, None)

    def get_all_trials(self):
        return self.trials.values()

    def get_trial_count(self):
        return len(self.trials)

    def add_feature_to_trials(self, f_name, f_values, feature_set):
        self.add_feature_name(f_name, feature_set)
        for trial_id in f_values:
            self.trials.get(trial_id).add_feature(f_name, f_values[trial_id])

    def add_valued_pattern(self, pattern):
        self.valued_patterns.append(pattern)

    def get_user_characteristics(self):
        return self.user_characteristics

    def set_user_characteristics(self, user_characteristics):
        self.user_characteristics = user_characteristics


class Trial(object):
    def __init__(self, pid, tid):
        self.pid = pid
        self.tid = tid
        self.scanpath = None
        self.aoivisits = []
        self.fixations = []
        self.features = {}
        self.pattern_frequencies = {}

    def __repr__(self):
        return "P{pid}, {tid}: {f} features".format(pid=self.pid, tid=self.tid, f=len(self.features))

    def get_trial_id(self):
        return self.pid, self.tid

    def get_pid(self):
        if self.pid.endswith('b'):
            return self.pid[:-1]
        return self.pid

    def get_tid(self):
        return self.tid

    def add_feature(self, f_name, f_value):
        if f_name not in self.features:
            self.features[f_name] = float(f_value)

    def get_feature(self, f_name):
        return self.features.get(f_name, None)

    def set_scanpath(self, scanpath):
        if scanpath:
            self.scanpath = scanpath.strip().split(' ')

    def get_scanpath(self):
        return self.scanpath

    def set_pattern_frequencies(self, pattern_frequencies):
        self.pattern_frequencies = pattern_frequencies

    def get_pattern_frequencies(self):
        return self.pattern_frequencies

    def get_pattern_frequency(self, pattern):
        return self.pattern_frequencies.get(pattern, 0)

    def add_aoi_visit(self, aoi_visit):
        self.aoivisits.append(aoi_visit)

    def add_fixation(self, fixation):
        self.fixations.append(fixation)

    def get_fixations(self):
        return self.fixations

    def get_long_visits(self, l=10):
        return sorted(sorted(self.aoivisits, key=lambda x: x.duration, reverse=True)[:l])


class AOIVisit(object):
    def __init__(self, abs_time, trial_time, aoi_name, duration):
        self.abs_time = int(abs_time)
        self.trial_time = int(trial_time)
        self.aoi_name = aoi_name
        self.duration = int(duration)

    def __repr__(self):
        return "{trial_time}: {aoi}({duration})".format(trial_time=self.trial_time,
                                                        aoi=self.aoi_name, duration=self.duration)

    def __cmp__(self, other):
        if hasattr(other, 'trial_time'):
            return self.trial_time.__cmp__(other.trial_time)

    def get_aoi_name(self):
        return self.aoi_name


class Fixation(object):
    def __init__(self, abs_time, trial_time, aoi_name, duration):
        self.abs_time = int(abs_time)
        self.trial_time = int(trial_time)
        self.aoi_name = aoi_name
        self.duration = int(duration)

    def __repr__(self):
        return "{trial_time}: {aoi}({duration})".format(trial_time=self.trial_time,
                                                        aoi=self.aoi_name, duration=self.duration)

    def __cmp__(self, other):
        if hasattr(other, 'trial_time'):
            return self.trial_time.__cmp__(other.trial_time)

    def get_aoi_name(self):
        return self.aoi_name


class Participant(object):
    def __init__(self, pid, ps=None, visual=None, verbal=None, locus_of_control=None, expertise=None):
        self.pid = pid
        self.trials = {}
        self.characteristic_values = {}
        self.characteristic_labels = {}
        self.set_characteristic_value('ps', ps)
        self.set_characteristic_value('visual', visual)
        self.set_characteristic_value('verbal', verbal)
        self.set_characteristic_value('loc', locus_of_control)
        self.set_characteristic_value('expertise', expertise)

    def get_id(self):
        return self.pid

    def get_characteristic_value(self, characteristic):
        return self.characteristic_values.get(characteristic)

    def set_characteristic_value(self, characteristic, value):
        if value:
            self.characteristic_values[characteristic] = float(value)

    def get_characteristic_label(self, characteristic):
        return self.characteristic_labels.get(characteristic)

    def set_characteristic_label(self, characteristic, label):
        self.characteristic_labels[characteristic] = label

    def add_trial(self, trial):
        task_id = trial.get_tid()
        if task_id not in self.trials:
            self.trials[task_id] = trial


class Task(object):
    def __init__(self, tid, difficulty=None):
        self.tid = tid
        self.difficulty = difficulty

    def get_id(self):
        return self.tid

    def get_difficulty(self):
        return self.difficulty


class ValuedPattern(object):
    """
    For valued pattern in annotated sequences, e.g., annotate an AOI sequence with fixation durations.
    """
    def __init__(self, items):
        self.items = items

    def __repr__(self):
        items = ""
        for item in self.items:
            items += "{aoi} ({min},{max})\t".format(aoi=item.aoi, min=item.min_duration, max=item.max_duration)
        return items

    def get(self, i):
        return self.items[i]

    def get_length(self):
        return len(self.items)


class ValuedItem(object):
    def __init__(self, aoi, min_duration, max_duration):
        self.aoi = aoi
        self.min_duration = int(min_duration)
        self.max_duration = int(max_duration)
