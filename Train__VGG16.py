import os
from glob import glob

import cv2
import numpy as np

from keras import preprocessing
from tensorflow.keras.utils import to_categorical

from tensorflow.keras.utils import plot_model

# get the reference to the webcam
width  = 32
height = 32

##############

def load_images(base_path):
    images = []
    path = os.path.join(base_path, '*.png')
    for image_path in glob(path):
        image = preprocessing.image.load_img(image_path,
                                             target_size=(width, height))
        x = preprocessing.image.img_to_array(image)

        images.append(x)
    return images



###############

a1 = np.array(load_images('./DevanagariHandwrittenCharacterDataset/Train/character_1_ka'))
a2 = np.array(load_images('./DevanagariHandwrittenCharacterDataset/Train/character_2_kha'))
a3 = np.array(load_images('./DevanagariHandwrittenCharacterDataset/Train/character_3_ga'))
a4 = np.array(load_images('./DevanagariHandwrittenCharacterDataset/Train/character_4_gha'))
a5 = np.array(load_images('./DevanagariHandwrittenCharacterDataset/Train/character_5_kna'))
a6 = np.array(load_images('./DevanagariHandwrittenCharacterDataset/Train/character_6_cha'))
a7 = np.array(load_images('./DevanagariHandwrittenCharacterDataset/Train/character_7_chha'))
a8 = np.array(load_images('./DevanagariHandwrittenCharacterDataset/Train/character_8_ja'))
a9 = np.array(load_images('./DevanagariHandwrittenCharacterDataset/Train/character_9_jha'))
a10 = np.array(load_images('./DevanagariHandwrittenCharacterDataset/Train/character_10_yna'))
print("10 folders loaded")
a11 = np.array(load_images('./DevanagariHandwrittenCharacterDataset/Train/character_11_taamatar'))
a12 = np.array(load_images('./DevanagariHandwrittenCharacterDataset/Train/character_12_thaa'))
a13 = np.array(load_images('./DevanagariHandwrittenCharacterDataset/Train/character_13_daa'))
a14 = np.array(load_images('./DevanagariHandwrittenCharacterDataset/Train/character_14_dhaa'))
a15 = np.array(load_images('./DevanagariHandwrittenCharacterDataset/Train/character_15_adna'))
a16 = np.array(load_images('./DevanagariHandwrittenCharacterDataset/Train/character_16_tabala'))
a17 = np.array(load_images('./DevanagariHandwrittenCharacterDataset/Train/character_17_tha'))
a18 = np.array(load_images('./DevanagariHandwrittenCharacterDataset/Train/character_18_da'))
a19 = np.array(load_images('./DevanagariHandwrittenCharacterDataset/Train/character_19_dha'))
a20 = np.array(load_images('./DevanagariHandwrittenCharacterDataset/Train/character_20_na'))
print("20 folders loaded")
a21 = np.array(load_images('./DevanagariHandwrittenCharacterDataset/Train/character_21_pa'))
a22 = np.array(load_images('./DevanagariHandwrittenCharacterDataset/Train/character_22_pha'))
a23 = np.array(load_images('./DevanagariHandwrittenCharacterDataset/Train/character_23_ba'))
a24 = np.array(load_images('./DevanagariHandwrittenCharacterDataset/Train/character_24_bha'))
a25 = np.array(load_images('./DevanagariHandwrittenCharacterDataset/Train/character_25_ma'))
a26 = np.array(load_images('./DevanagariHandwrittenCharacterDataset/Train/character_26_yaw'))
a27 = np.array(load_images('./DevanagariHandwrittenCharacterDataset/Train/character_27_ra'))
a28 = np.array(load_images('./DevanagariHandwrittenCharacterDataset/Train/character_28_la'))
a29 = np.array(load_images('./DevanagariHandwrittenCharacterDataset/Train/character_29_waw'))
a30 = np.array(load_images('./DevanagariHandwrittenCharacterDataset/Train/character_30_motosaw'))
print("30 folders loaded")
a31 = np.array(load_images('./DevanagariHandwrittenCharacterDataset/Train/character_31_petchiryakha'))
a32 = np.array(load_images('./DevanagariHandwrittenCharacterDataset/Train/character_32_patalosaw'))
a33 = np.array(load_images('./DevanagariHandwrittenCharacterDataset/Train/character_33_ha'))
a34 = np.array(load_images('./DevanagariHandwrittenCharacterDataset/Train/character_34_chhya'))
a35 = np.array(load_images('./DevanagariHandwrittenCharacterDataset/Train/character_35_tra'))
a36 = np.array(load_images('./DevanagariHandwrittenCharacterDataset/Train/character_36_gya'))
a37 = np.array(load_images('./DevanagariHandwrittenCharacterDataset/Train/chma'))
a38 = np.array(load_images('./DevanagariHandwrittenCharacterDataset/Train/chya'))
a39 = np.array(load_images('./DevanagariHandwrittenCharacterDataset/Train/digit_0'))
a40 = np.array(load_images('./DevanagariHandwrittenCharacterDataset/Train/digit_1'))
print("40 folders loaded")
a41 = np.array(load_images('./DevanagariHandwrittenCharacterDataset/Train/digit_2'))
a42 = np.array(load_images('./DevanagariHandwrittenCharacterDataset/Train/digit_3'))
a43 = np.array(load_images('./DevanagariHandwrittenCharacterDataset/Train/digit_4'))
a44 = np.array(load_images('./DevanagariHandwrittenCharacterDataset/Train/digit_5'))
a45 = np.array(load_images('./DevanagariHandwrittenCharacterDataset/Train/digit_6'))
a46 = np.array(load_images('./DevanagariHandwrittenCharacterDataset/Train/digit_7'))
a47 = np.array(load_images('./DevanagariHandwrittenCharacterDataset/Train/digit_8'))
a48 = np.array(load_images('./DevanagariHandwrittenCharacterDataset/Train/digit_9'))
a49 = np.array(load_images('./DevanagariHandwrittenCharacterDataset/Train/mya'))
a50 = np.array(load_images('./DevanagariHandwrittenCharacterDataset/Train/shva'))
a51 = np.array(load_images('./DevanagariHandwrittenCharacterDataset/Train/swa'))
a52 = np.array(load_images('./DevanagariHandwrittenCharacterDataset/Train/tva'))
print("52 folders loaded")





X = np.concatenate((a1,a2,a3,a4,a5,a6,a7,a8,a9,a10,a11,a12,a13,a14,a15,a16,a17,a18,a19,a20,a21,a22,a23,a24,a25,a26,a27,a28,a29,a30
                    ,a31,a32,a33,a34,a35,a36,a37,a38,a39,a40,a41,a42,a43,a44,a45,a46,a47,a48,a49,a50,a51,a52), axis=0)

##############

# normalization
X = X / 255.

###################


ya1 = [0 for item in enumerate(a1)]
ya2 = [1 for item in enumerate(a2)]
ya3 = [2 for item in enumerate(a3)]
ya4 = [3 for item in enumerate(a4)]
ya5 = [4 for item in enumerate(a5)]
ya6 = [5 for item in enumerate(a6)]
ya7 = [6 for item in enumerate(a7)]
ya8 = [7 for item in enumerate(a8)]
ya9 = [8 for item in enumerate(a9)]

ya10 = [9 for item in enumerate(a10)]
ya11 = [10 for item in enumerate(a11)]
ya12 = [11 for item in enumerate(a12)]
ya13 = [12 for item in enumerate(a13)]
ya14 = [13 for item in enumerate(a14)]
ya15 = [14 for item in enumerate(a15)]
ya16 = [15 for item in enumerate(a16)]
ya17 = [16 for item in enumerate(a17)]
ya18 = [17 for item in enumerate(a18)]
ya19 = [18 for item in enumerate(a19)]

ya20 = [19 for item in enumerate(a20)]
ya21 = [20 for item in enumerate(a21)]
ya22 = [21 for item in enumerate(a22)]
ya23 = [22 for item in enumerate(a23)]
ya24 = [23 for item in enumerate(a24)]
ya25 = [24 for item in enumerate(a25)]
ya26 = [25 for item in enumerate(a26)]
ya27 = [26 for item in enumerate(a27)]
ya28 = [27 for item in enumerate(a28)]
ya29 = [28 for item in enumerate(a29)]

ya30 = [29 for item in enumerate(a30)]
ya31 = [30 for item in enumerate(a31)]
ya32 = [31 for item in enumerate(a32)]
ya33 = [32 for item in enumerate(a33)]
ya34 = [33 for item in enumerate(a34)]
ya35 = [34 for item in enumerate(a35)]
ya36 = [35 for item in enumerate(a36)]
ya37 = [36 for item in enumerate(a37)]
ya38 = [37 for item in enumerate(a38)]
ya39 = [38 for item in enumerate(a39)]

ya40 = [39 for item in enumerate(a40)]
ya41 = [40 for item in enumerate(a41)]
ya42 = [41 for item in enumerate(a42)]
ya43 = [42 for item in enumerate(a43)]
ya44 = [43 for item in enumerate(a44)]
ya45 = [44 for item in enumerate(a45)]
ya46 = [45 for item in enumerate(a46)]
ya47 = [46 for item in enumerate(a47)]
ya48 = [47 for item in enumerate(a48)]
ya49 = [48 for item in enumerate(a49)]

ya50 = [49 for item in enumerate(a50)]
ya51 = [50 for item in enumerate(a51)]
ya52 = [51 for item in enumerate(a52)]



y = np.concatenate((ya1,ya2,ya3,ya4,ya5,ya6,ya7,ya8,ya9,ya10,ya11,ya12,ya13,ya14,ya15,ya16,ya17,ya18,ya19,ya20
                    ,ya21,ya22,ya23,ya24,ya25,ya26,ya27,ya28,ya29,ya30,ya31,ya32,ya33,ya34,ya35,ya36,ya37,ya38,ya39,ya40
                    ,ya41,ya42,ya43,ya44,ya45,ya46,ya47,ya48,ya49,ya50,ya51,ya52), axis=0)

y = to_categorical(y, num_classes=52)

#print(y.shape)


#####################
from keras.models import Sequential
from keras.layers.core import Activation, Dropout, Flatten, Dense
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import adam_v2

# default parameters
conv_1 = 16
conv_1_drop = 0.2
conv_2 = 32
conv_2_drop = 0.2
dense_1_n = 1024
dense_1_drop = 0.2
dense_2_n = 512
dense_2_drop = 0.2
lr = 0.001

epochs = 500
batch_size = 32
color_channels = 3

def build_model(conv_1_drop=conv_1_drop, conv_2_drop=conv_2_drop,
                dense_1_n=dense_1_n, dense_1_drop=dense_1_drop,
                dense_2_n=dense_2_n, dense_2_drop=dense_2_drop,
                lr=lr):
    model = Sequential()

# 1st and 2nd Convolutional Layer
    model.add(Convolution2D(64, (3, 3),
                            input_shape=(width, height, color_channels),
                            activation='relu',padding="same"))
    model.add(Convolution2D(64, (3, 3), activation='relu',padding="same"))
    model.add(MaxPooling2D(pool_size=(2, 2),strides=(2,2)))

# 3rd and 4th Convolutional Layer
    model.add(Convolution2D(128, (3, 3), activation='relu',padding="same"))
    model.add(Convolution2D(128, (3, 3), activation='relu',padding="same"))
    model.add(MaxPooling2D(pool_size=(2, 2),strides=(2,2)))

# 5th, 6th and 7th Convolutional Layer
    model.add(Convolution2D(128, (3, 3), activation='relu',padding="same"))
    model.add(Convolution2D(128, (3, 3), activation='relu',padding="same"))
    model.add(Convolution2D(128, (3, 3), activation='relu',padding="same"))
    model.add(MaxPooling2D(pool_size=(2, 2),strides=(2,2)))

# 8th, 9th and 10th Convolutional Layer
    model.add(Convolution2D(256, (3, 3), activation='relu',padding="same"))
    model.add(Convolution2D(256, (3, 3), activation='relu',padding="same"))
    model.add(Convolution2D(256, (3, 3), activation='relu',padding="same"))
    model.add(MaxPooling2D(pool_size=(2, 2),strides=(2,2)))

# 11th, 12th and 13th Convolutional Layer
    model.add(Convolution2D(512, (3, 3), activation='relu',padding="same"))
    model.add(Convolution2D(512, (3, 3), activation='relu',padding="same"))
    model.add(Convolution2D(512, (3, 3), activation='relu',padding="same"))
    model.add(MaxPooling2D(pool_size=(2, 2),strides=(2,2)))

# Passing it to a dense layer        
    model.add(Flatten())

# 1st Dense Layer    
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.4))

# 2nd Dense Layer
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.4))

# 3rd Dense Layer
    model.add(Dense(52, activation='softmax'))

    model.compile(loss='categorical_crossentropy',
                  optimizer=adam_v2.Adam(learning_rate=lr),
                  metrics=['accuracy'])

    return model

#######################


# model with base parameters
model = build_model()

model.summary()


plot_model(model, to_file='model.png', show_shapes=True, show_layer_names=True,  expand_nested=False)

#################
epochs = 100
##################

model.fit(X, y, epochs=epochs)

model.save('please.h5')




print("Model_Ready")



