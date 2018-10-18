# ClassicalMER-LongtermFeatures

This is an attempt to capture the emotional reaction of a listner to classical music pieces, for predicting the valence and arousal reactions of a perticular listner, using a regression approach. The pieces used in our dataset is from MusicNet (https://homes.cs.washington.edu/~thickstn/musicnet.html). Long Term Features were extracted using the pyAudioAnalysis library (https://github.com/tyiannak/pyAudioAnalysis), with a short-term window and step of 50 ms and 25 ms, and a mid-term window and step of 2s and 0.2s.

Our data was split into 3 catagories based on the audio features used : Temporal-Spectral-Rhythm features Dataset(330X70), Temporal-Spectral Features(330X68), With Fisher-Score Feature Selection on the Temporal-Spectral-Rhythm Dataset (330X60). 

The models used are Support Vector Regression, Random Forrest Regression, and an Artificail Nueural Network with 2 layers and 90 nodes in each layer. The dataset of was split into train and test (in an 80:20 ratio). On the training set 10 fold crossvalidation was applied and the hyperparameters were tuned using Grid Seach. 

The analysis is done on the dataset as well as the model used.
