# -*- coding: utf-8 -*-
"""


@author: SaiPr
"""

import numpy as np 


#Audio files should be of the type .wav 
# This is for testing new files 
mfcc_val = wavtomfcc('path of file ',20,11)
#reshaping to fit the classifier
mfcc_val = mfcc_val.reshape(1,20,11,1)
#predecting
pred  = classifier.predict(mfcc)
print(pred)