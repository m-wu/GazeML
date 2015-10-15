import sequence

_author__ = 'Mike Wu'


def add_long_aoi_visit_features(exp):
    """
    Query the AOI sequences with the l longest AOI visits,
    extract patterns from these sequences as features,
    and compute the features for each trial.
    """
    long_aoivisit_sequences = []
    for trial in exp.get_all_trials():
        long_aoivisit_sequence = trial.get_long_visits(l=5)
        long_aoivisit_sequences.append(get_aoi_visits_string(long_aoivisit_sequence, " "))

    long_visit_patterns = sequence.get_common_patterns(long_aoivisit_sequences, 25, length_min=2, length_max=5)

    for p in long_visit_patterns:
        pattern_counts = {}
        for trial in exp.get_all_trials():
            aoivisit_sequence = trial.get_long_visits(l=5)
            sequence_string = get_aoi_visits_string(aoivisit_sequence, "")
            pattern_count = sequence.count_pattern_occurrence(sequence_string, p)
            pattern_counts[trial.get_trial_id()] = pattern_count
        exp.add_feature_to_trials(p, pattern_counts, "long_aoi")


def add_valued_pattern_features(exp):
    for pattern in exp.valued_patterns:
        pattern_counts = {}
        for trial in exp.get_all_trials():
            pattern_count = sequence.count_valued_pattern_occurrence(trial.get_fixations(), pattern)
            pattern_counts[trial.get_trial_id()] = pattern_count
        exp.add_feature_to_trials(pattern, pattern_counts, "valued_pattern")


def populate_sequence_pattern_frequency(exp):
    for trial in exp.get_all_trials():
        scanpath = trial.get_scanpath()
        trial.set_pattern_frequencies(sequence.get_pattern_frequency_single_sequence(scanpath, 3, 7))


def get_sequence_pattern_features(exp, class_name, trial_index, s_support=0, alpha=0.05):
    # separate sequences into two groups
    pattern_frequency_list_0 = []
    pattern_frequency_list_1 = []
    for trial in [exp.get_all_trials()[i] for i in trial_index]:
        pattern_frequencies = trial.get_pattern_frequencies()
        if get_target_label(exp, trial, class_name) == 0:
            pattern_frequency_list_0.append(pattern_frequencies)
        else:
            pattern_frequency_list_1.append(pattern_frequencies)

    n_trials_0 = len(pattern_frequency_list_0)
    n_trials_1 = len(pattern_frequency_list_1)

    pattern_frequencies_0 = sequence.consolidate_pattern_frequency_list(pattern_frequency_list_0)
    pattern_frequencies_1 = sequence.consolidate_pattern_frequency_list(pattern_frequency_list_1)
    selected_patterns = set(pattern_frequencies_0.keys()) | set(pattern_frequencies_1.keys())

    if s_support > 0:
        # get frequent patterns for each group and merge them
        frequent_patterns_0 = sequence.get_frequent_patterns(pattern_frequencies_0, s_support * n_trials_0)
        frequent_patterns_1 = sequence.get_frequent_patterns(pattern_frequencies_1, s_support * n_trials_1)
        selected_patterns = set(frequent_patterns_0) | set(frequent_patterns_1)

    # filter patterns by significant difference in frequency
    if alpha < 1:
        selected_patterns = sequence.filter_significant_patterns(selected_patterns, pattern_frequencies_0,
                                                                 pattern_frequencies_1, n_trials_0, n_trials_1, alpha)

    # count pattern occurrences and add them as features
    x = []
    for trial in exp.get_all_trials():
        pattern_counts = []
        for pattern in selected_patterns:
            pattern_counts.append(trial.get_pattern_frequency(pattern))
        x.append(pattern_counts)

    return x


def get_aoi_visits_string(aoivisits, separator):
    return separator.join([aoivisit.get_aoi_name() for aoivisit in aoivisits])


def get_fixations_string(fixations, separator):
    return separator.join([fixation.get_aoi_name() for fixation in fixations])


def get_target_label(exp, trial, class_name):
    if class_name in exp.get_user_characteristics():
        return exp.get_participant(trial.get_pid()).get_characteristic_label(class_name)
    return None
