import os
import h5py
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Convolution2D,MaxPooling2D,Activation, Dropout, Flatten, Dense, BatchNormalization


trainDataGen = ImageDataGenerator(
		rotation_range = 5,
		width_shift_range = 0.1,
		height_shift_range = 0.1,
		rescale = 1.0/255,
		shear_range = 0.2,
		zoom_range = 0.2,		
		horizontal_flip = False,
		fill_mode = 'nearest')

test_datagen = ImageDataGenerator(rescale=1./255)
trainGenerator = trainDataGen.flow_from_directory(
			"DevanagariHandwrittenCharacterDataset/Train",
			target_size = (227,227),
			batch_size = 32,
			color_mode = "grayscale",
			class_mode = "categorical")
prev = ""
labels = ["ka","kha","ga","gha","kna","cha","chha","ja","jha","yna","t`a","t`ha","d`a","d`ha","adna","ta","tha","da","dha","na","pa","pha","ba","bha","ma","yaw","ra","la","waw","sha","shat","sa","ha","aksha","tra","gya","chma","chya","0","1","2","3","4","5","6","7","8","9","mya","shva","swa","tva"]
count = 0;

validation_generator = test_datagen.flow_from_directory(
			"DevanagariHandwrittenCharacterDataset/Test",
			target_size=(227,227),
			batch_size=32,
			color_mode = "grayscale",
			class_mode= 'categorical')

	
model = Sequential()

#Layer1----------------------------------------------------------
model.add(Convolution2D(filters = 96,
			kernel_size = (11,11),
			strides = 4,
			activation = "relu",
			input_shape = (227,227,1)))

model.add(BatchNormalization())
#model.add(Dropout(0.25))
model.add(MaxPooling2D(pool_size=(3,3),
			strides=(2, 2)))

#Layer2-------------------------------------------------------------
model.add(Convolution2D(filters = 256,
			kernel_size = (5,5),
			strides = 1,
			activation = "relu",
			padding="same"))
model.add(BatchNormalization())
#model.add(MaxPooling2D())
model.add(MaxPooling2D(pool_size=(3,3),
			strides=(2, 2),
			))
#model.add(Dropout(0.25))


#Layers 3-----------------------------------------------------------	
model.add(Convolution2D(filters = 384,
			kernel_size = (3,3),
			strides = 1,
			activation = "relu",
             padding="same"))

model.add(Convolution2D(filters = 384,
			kernel_size = (3,3),
			strides = 1,
			activation = "relu",
             padding="same"))

model.add(Convolution2D(filters = 256,
			kernel_size = (3,3),
			strides = 1,
			activation = "relu",
             padding="same"))

model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(3, 3),
			strides=(2, 2)))


			
model.add(Flatten())

model.add(Dense(4096,
		activation = "relu"))
model.add(Dropout(0.5))			

model.add(Dense(4096,
		activation = "relu"))
model.add(Dropout(0.5))			

model.add(Dense(52,
		activation = "softmax"))			
			
model.compile(optimizer = "adam",
		loss = "categorical_crossentropy",
		metrics = ["accuracy"])
		
print(model.summary())

val_step = int(np.ceil(validation_generator.samples/32))
step = int(np.ceil(trainGenerator.samples/32))

res=model.fit_generator(
		trainGenerator,
		epochs = 120,
		steps_per_epoch = 52,
		validation_data = validation_generator,
		validation_steps = 52
		)

model.save("Alexnet1.h5")


print('model Ready')       

##%matplotlib inline
accu=res.history['accuracy']
val_acc=res.history['val_accuracy']
loss=res.history['loss']
val_loss=res.history['val_loss']

epochs=range(len(accu)) #No. of epochs

import matplotlib.pyplot as plt
plt.plot(epochs,accu,'r',label='Training Accuracy')
plt.plot(epochs,val_acc,'g',label='Testing Accuracy')
plt.legend()
plt.figure()
##
###Plot training and validation loss per epoch
plt.plot(epochs,loss,'r',label='Training Loss')
plt.plot(epochs,val_loss,'g',label='Testing Loss')
plt.legend()
plt.show()
		

