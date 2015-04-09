class Experiment(object):
    def __init__(self):
        self.trials = {}
        self.participants = {}
        self.tasks = {}
        self.feature_names = []

    def add_participant(self, participant):
        p_id = participant.get_id()
        if p_id not in self.participants:
            self.participants[participant.get_id()] = participant

    def get_participant(self, pid):
        return self.participants.get(pid, None)

    def get_all_participants(self):
        return self.participants.values()

    def add_task(self, task):
        task_id = task.get_id()
        if task_id not in self.tasks:
            self.tasks[task.get_id()] = task

    def add_feature_name(self, feature_name):
        self.feature_names.append(feature_name)

    def add_feature_names(self, feature_names):
        self.feature_names.extend(feature_names)

    def get_feature_names(self):
        return self.feature_names

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

    def add_feature_to_trials(self, f_name, f_values):
        self.add_feature_name(f_name)
        for trial_id in f_values:
            self.trials.get(trial_id).add_feature(f_name, f_values[trial_id])


class Trial(object):
    def __init__(self, pid, tid):
        self.pid = pid
        self.tid = tid
        self.scanpath = None
        self.aoivisits = []
        self.features = {}

    def __repr__(self):
        return "P{pid}, {tid}: {f} features".format(pid=self.pid, tid=self.tid, f=len(self.features))

    def get_trial_id(self):
        return (self.pid, self.tid)

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

    def add_scanpath(self, scanpath):
        self.scanpath = scanpath

    def get_scanpath(self):
        return self.scanpath

    def add_aoivisit(self, aoivisit):
        self.aoivisits.append(aoivisit)

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


class Participant(object):
    def __init__(self, pid, ps=None, visual=None, verbal=None):
        self.pid = pid
        self.ps = float(ps)
        self.visual = float(visual)
        self.verbal = float(verbal)

    def get_id(self):
        return self.pid

    def get_ps(self):
        return self.ps

    def get_visual(self):
        return self.visual

    def get_verbal(self):
        return self.verbal


class Task(object):
    def __init__(self, tid, difficulty=None):
        self.tid = tid
        self.difficulty = difficulty

    def get_id(self):
        return self.tid

    def get_difficulty(self):
        return self.difficulty