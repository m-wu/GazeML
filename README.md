# GazeML

GazeML is a python application that runs machine learning experiment to test the classification of user characteristics from eye-tracking data.

GazeML reads in existing features generated from eye-tracking data in an experiment (e.g., by using [EMDAT](https://github.com/atuav/emdat)) and runs cross validations to test the performance of different feature sets and classifiers. It is built on the scikit-learn machine learning library.

When given gaze movement sequence data, GazeML can extract patterns from the sequences and compute features based on the frequency of the patterns.
