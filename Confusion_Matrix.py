import os
from glob import glob
import cv2

import numpy as np
import argparse
import imutils
import cv2

"""

import cv2
import time
import numpy as np

from PIL import Image
from datetime import datetime

import time

import os
import shutil

from keras.preprocessing.image import img_to_array
from keras.models import load_model

import numpy as np
from keras.preprocessing import image
import tensorflow as tf

model = tf.keras.models.load_model("MarathiModel_3.h5")


directory = os.getcwd()
persons = glob(directory+'/data/*')

##persons = glob(directory+'/DevanagariHandwrittenCharacterDataset/Train/*')

i = 0
true_label = []
predicted_label = []

for person_name in persons:
    Known_names = os.path.basename(person_name)
    path = os.path.join(person_name, '*.png')
    print(path)    
    for img in glob(path):
        true_label.append(i)
        test_image = cv2.imread(img)
        image = cv2.resize(test_image, (32,32))
        image = image.astype("float") / 255.0
        image = img_to_array(image)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = np.expand_dims(image, axis=0)
        image = np.expand_dims(image, axis=3)

        lists = model.predict(image)[0]
        k = int(np.argmax(lists))
        predicted_label.append(k)

    i += 1


print(true_label)
print("======")
print(predicted_label)

"""


labels = [u'\u091E',u'\u091F',u'\u0920',u'\u0921',u'\u0922',u'\u0923',u'\u0924',u'\u0925',u'\u0926',u'\u0927',u'\u0915',u'\u0928',u'\u092A',u'\u092B',u'\u092c',u'\u092d',u'\u092e',u'\u092f',u'\u0930',u'\u0932',u'\u0935',u'\u0916',u'\u0936',u'\u0937',u'\u0938',u'\u0939','क्ष','त्र','ज्ञ','च्म','च्या',u'\u0917',u'\u0918',u'\u0919',u'\u091a',u'\u091b',u'\u091c',u'\u091d',u'\u0966',u'\u0967',u'\u0968',u'\u0969',u'\u096a',u'\u096b',u'\u096c',u'\u096d',u'\u096e',u'\u096f','म्या','श्व','स्व','त्व']


print(len(labels))

21  :  ख => 22nd 
22  :  श
23  :  ष
24  :  स => 25 component in array
25  :  ह
26  :  क्ष
27  :  त्र
28  :  ज्ञ
29  :  च्म
30  :  च्या
31  :  ग
32  :  घ
33  :  ङ
34  :  च
35  :  छ
36  :  ज
37  :  झ
38  :  ०
39  :  १
40  :  २
41  :  ३
42  :  ४
43  :  ५
44  :  ६
45  :  ७
46  :  ८
47  :  ९
48  :  म्या
49  :  श्व
50  :  स्व
51  :  त्व

pp = [36,12,13,14,15,   16,17,18,19,20,     0,21,22,23,24,      25,26,27,28,29,
      30,1,31 ,32,33,      34,35,36,37,38,      39,2,3,4,5,    6,7,8,40,41,
      42,43,44,45,46,     47,48,49,50,51]

#Initialize array     
arr = [1, 2, 3, 4, 2, 7, 8, 8, 3];     


arr = pp     
print("Duplicate elements in given array: ");    
#Searches for duplicate element    
for i in range(0, len(arr)):    
    for j in range(i+1, len(arr)):    
        if(arr[i] == arr[j]):    
            print(arr[j]);   


##print(len(pp))
##
####i = 0
####for k in labels:
####    print(str(i) + "  :  " + k)
####    i += 1
####
####print()







