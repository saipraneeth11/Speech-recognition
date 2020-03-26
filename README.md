# Speech-recognition
A word recognition programm  created based on deep learning and DSP

# Data 
I have obtained the data from tensor flow 's speech rec file 

https://storage.cloud.google.com/download.tensorflow.org/data/speech_commands_v0.01.tar.gz

This is a pretty large file containing audio samples of around 30 words 

For this project I have used the words one,two,three,four,five

# Processing of Data
I have converted the audio data(amplitudes) to MFCC features that can be feed to a CNN.

A brief explanation of MFCC

MFCC stands for Mel Frequency Cepstral Co-efficients.

MFCC is obtained by the following steps 

1. Taking the fourier transform of time domain signal.

2. This spectrum is taken on to a log scale 

3. To this we apply a cosine transform 

 Ta Da ! we have obtained MFCC 
 
 This spectrum is called cepstrum !
 
 # Making the CNN 
 
 The goal is to classify the words into five classes
 
 The CNN contains 
 1. Two convolution layers with relu activation 
 2. Dropout feature optional (To avoid overfitting)
 3. A fully connected layer 
 
 We can use a RNN also for this case or even a simple regression model also!
 
 # Observation 
 
Using the CNN , I have obtained a accuracy of 98.23 
