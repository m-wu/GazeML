from collections import Counter, defaultdict
from scipy import stats

_author__ = 'Mike Wu'


def get_common_patterns(sequences, most_common, length_min=3, length_max=7):
    patterns = []
    for length in range(length_min, length_max+1):      
        for sequence in sequences:
            patterns.extend(get_ngrams(sequence, length))

    return [i[0] for i in Counter(patterns).most_common(most_common)]


def consolidate_pattern_frequency_list(pattern_frequency_list):
    pattern_frequencies = defaultdict(list)
    for trial_pattern_frequencies in pattern_frequency_list:
        for pattern in trial_pattern_frequencies:
            pattern_frequencies[pattern].append(trial_pattern_frequencies[pattern])
    return pattern_frequencies


def get_frequent_patterns(pattern_frequencies, min_sequence_appearance):
    """
    Return patterns that appear in at least a certain number of sequences.
    :param pattern_frequencies: a dictionary of patterns with their list of appearances per trial
    :param min_sequence_appearance: minimum number of sequences in which the pattern must appear
    :return: a list of patterns that appears in at least min_sequence_appearance number of sequences
    """
    pattern_appearances = {p: len(pattern_frequencies[p]) for p in pattern_frequencies}
    return [p for p in pattern_appearances if pattern_appearances[p] > min_sequence_appearance]


def filter_significant_patterns(patterns, pattern_frequencies_0, pattern_frequencies_1, n_trials_0, n_trials_1, alpha=0.05):
    pattern_appearances_0 = {p: len(pattern_frequencies_0[p]) for p in pattern_frequencies_0}
    pattern_appearances_1 = {p: len(pattern_frequencies_1[p]) for p in pattern_frequencies_1}
    # get all patterns that appear in either sequence
    significant_patterns = []
    for pattern in patterns:
        cont_table = [[pattern_appearances_0.get(pattern, 0), n_trials_0 - pattern_appearances_0.get(pattern, 0)],
                      [pattern_appearances_1.get(pattern, 0), n_trials_1 - pattern_appearances_1.get(pattern, 0)]]
        (chi2, ss_p_value) = stats.chi2_contingency(cont_table)[:2]

        if ss_p_value < alpha:
            significant_patterns.append(pattern)
            continue

        # build two lists of pattern occurrence per sequence, including 0 if the pattern is absent.
        a = pattern_frequencies_0.get(pattern, [])
        b = pattern_frequencies_1.get(pattern, [])
        a_full = a + [0] * (n_trials_0 - len(a))
        b_full = b + [0] * (n_trials_1 - len(b))

        # get p-value for the t-test.
        apf_p_value = stats.ttest_ind(a_full, b_full, equal_var=False)[1]

        if apf_p_value < alpha:
            significant_patterns.append(pattern)

    return significant_patterns


def get_pattern_frequency_single_sequence(sequence, length_min, length_max):
    pattern_frequencies = {}
    for length in range(length_min, length_max + 1):
        pattern_frequencies.update(get_ngram_frequencies(sequence, length))
    return pattern_frequencies


def get_pattern_frequency_multiple_sequences(sequences, length_min, length_max):
    """ Extract patterns in sequences and store the pattern frequencies per sequence
    :param sequences: the list of sequences from which to extract the patterns
    :param length_min: minimum pattern length
    :param length_max: maximum pattern length
    :return: a dictionary with pattern as the key and a list of number of pattern occurrences per sequence as the value;
    a sequence with certain pattern does not appear as 0 in the list of frequencies.
    """
    pattern_frequencies_all = {}
    for length in range(length_min, length_max + 1):
        for sequence in sequences:
            patterns = get_ngrams(sequence, length)
            pattern_frequencies = Counter(patterns)
            for pattern in pattern_frequencies.keys():
                pattern_frequencies_all.setdefault(pattern, []).append(pattern_frequencies[pattern])
    return pattern_frequencies_all


def get_ngrams(sequence, ngram_length):
    """ Output n-grams of length n from words in sequence"""
    n_grams = []
    for i in range(len(sequence) - ngram_length + 1):
        n_grams.append(tuple(sequence[i:i + ngram_length]))
    return n_grams


def get_ngram_frequencies(sequence, n):
    n_grams_frequencies = defaultdict(int)
    for i in range(len(sequence) - n + 1):
        n_gram = tuple(sequence[i:i + n])
        n_grams_frequencies[n_gram] += 1
    return n_grams_frequencies


def count_pattern_occurrence(sequence, pattern):
    patterns = get_ngrams(sequence, len(pattern))
    pattern_frequencies = Counter(patterns)
    return pattern_frequencies[pattern]


def count_valued_pattern_occurrence(fixations, pattern):
    l = pattern.get_length()
    counter = 0
    for i in range(len(fixations)-l+1):
        sequence = iter(fixations[i:i+l])
        counter += match_sequence(pattern, sequence, 0)
    return counter


def match_sequence(pattern, sequence, curr_index):
    fixation = sequence.next()

    if not match_fixation(fixation, pattern.get(curr_index)):
        return False

    if (curr_index == pattern.get_length() - 1) or match_sequence(pattern, sequence, curr_index+1):
        return True

    return False


def match_fixation(fixation, pattern_item):
    return (fixation.aoi_name == pattern_item.aoi) and (pattern_item.min_duration <= fixation.duration <= pattern_item.max_duration)