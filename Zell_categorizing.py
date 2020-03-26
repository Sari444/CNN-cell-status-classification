# -*- coding: utf-8 -*-
"""
Created on Wed May 22 13:51:04 2019

@author: Sarah rudigkeit
"""

import json
import os


import pandas as pd
import numpy as np

from tqdm import tqdm
import random as rn


# Plotting
import matplotlib.pyplot as plt
#%matplotlib inline
import seaborn as sns


# Image Processing
import cv2
from PIL import Image


# SK-Learn
from sklearn.metrics import f1_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
#from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from keras.preprocessing.image import ImageDataGenerator


# KERAS, Tensorflow
from tensorflow.keras import backend as K
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam,SGD,Adagrad,Adadelta,RMSprop
from tensorflow.keras.utils import to_categorical

from tensorflow.keras.layers import Dropout, Flatten,Activation
from tensorflow.keras.layers import Conv2D, MaxPooling2D, BatchNormalization

from tensorflow.keras.callbacks import ModelCheckpoint, History
from tensorflow.keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

import tensorflow as tf

# set variables 

main_folder = r'C:\Users\Sarah rudigkeit\Documents\Sarah\PhD\Zellnachverfolgung\Training'

filename1 = os.path.join(main_folder, r'Juni 2019\Juni2019_Pos1_uberarbeitet.json')
filename2 = os.path.join(main_folder, r'Juni 2019\Juni2019_Pos2_uberarbeitet.json')
filename3 = os.path.join(main_folder, r'Juni 2019\Juni2019_Pos3_uberarbeitet.json')
filename4 = os.path.join(main_folder, r'Juni 2019\Juni2019_Pos4_uberarbeitet.json')
filename5 = os.path.join(main_folder, r'Oktober 2018\Pos1_2240+2241_uberarbeitet.json')
filename6 = os.path.join(main_folder, r'Oktober 2018\Pos2_2240+2241_uberarbeitet.json')
filename7 = os.path.join(main_folder, r'Oktober 2018\Pos3_2240+2241_uberarbeitet.json')
filename8 = os.path.join(main_folder, r'Oktober 2018\Pos4_2240+2241_uberarbeitet.json')


images_folder = os.path.join(main_folder, r'Images') 
#images_folder11 = os.path.join(main_folder, r'648_Pos1')
#images_folder12 = os.path.join(main_folder, r'649_Pos1')
#images_folder21 = os.path.join(main_folder, r'648_Pos2')
#images_folder22 = os.path.join(main_folder, r'648_Pos2')
#images_folder31 = os.path.join(main_folder, r'648_Pos3')
#images_folder32 = os.path.join(main_folder, r'648_Pos3')
#images_folder41 = os.path.join(main_folder, r'648_Pos4')
#images_folder42 = os.path.join(main_folder, r'648_Pos4')
#EXAMPLE_PIC = os.path.join(images_folder, '1-1-Hela_TMRE_2240_Pos1.jpg')

TRAINING_SAMPLES = 10000
VALIDATION_SAMPLES = 2000
TEST_SAMPLES = 2000
IMG_WIDTH = 178
IMG_HEIGHT = 218
BATCH_SIZE = 16
NUM_EPOCHS = 20
print(images_folder)
datastore1 = []
datastore2 = []
datastore3 = []
datastore4 = []
datastore5 = []
datastore6 = []
datastore7 = []
datastore8 = []

with open(filename1, 'r') as f:
    datastore1 = json.load(f)
    
with open(filename2, 'r') as f:
    datastore2 = json.load(f)
    
with open(filename3, 'r') as f:
    datastore3 = json.load(f)
    
with open(filename4, 'r') as f:
    datastore4 = json.load(f)

with open(filename5, 'r') as f:
    datastore5 = json.load(f)
    
with open(filename6, 'r') as f:
    datastore6 = json.load(f)

with open(filename7, 'r') as f:
    datastore7 = json.load(f)

with open(filename8, 'r') as f:
    datastore8 = json.load(f)
    
image_size = {'image_width': 1388, 'image_height': 1044}

cell_df = pd.DataFrame(columns = ['imagename', 'x', 'width', 'y', 'height', 'label'])


for item_images in list(datastore1['_via_img_metadata'].keys()):
    str_filename = datastore1['_via_img_metadata'][item_images]['filename']
    for item_regions in datastore1['_via_img_metadata'][item_images]['regions']:
            
        try:
            for keys in item_regions['region_attributes']['type']: 
                label = str(keys)
            
            x = item_regions['shape_attributes']['x']
            y = item_regions['shape_attributes']['y']
            width = item_regions['shape_attributes']['width']
            height = item_regions['shape_attributes']['height']
            
            it_df = pd.DataFrame({'imagename': [str_filename], 'x': [x], 'width': [width], 'y': [y], 'height': [height], 'label': [label]})
            
            cell_df = cell_df.append(it_df, ignore_index = True)
            
            
        except:
            pass

for item_images in list(datastore2['_via_img_metadata'].keys()):
    str_filename = datastore2['_via_img_metadata'][item_images]['filename']
    for item_regions in datastore2['_via_img_metadata'][item_images]['regions']:
            
        try:
            for keys in item_regions['region_attributes']['type']: 
                label = str(keys)
            #label = item_regions['region_attributes']['type']
            
            x = item_regions['shape_attributes']['x']
            y = item_regions['shape_attributes']['y']
            width = item_regions['shape_attributes']['width']
            height = item_regions['shape_attributes']['height']
            
            it_df = pd.DataFrame({'imagename': [str_filename], 'x': [x], 'width': [width], 'y': [y], 'height': [height], 'label': [label]})
            
            cell_df = cell_df.append(it_df, ignore_index = True)
            
            
        except:
            pass
        
for item_images in list(datastore3['_via_img_metadata'].keys()):
    str_filename = datastore3['_via_img_metadata'][item_images]['filename']
    for item_regions in datastore3['_via_img_metadata'][item_images]['regions']:
            
        try:
            for keys in item_regions['region_attributes']['type']: 
                label = str(keys)
                #label = item_regions['region_attributes']['type']
            
            x = item_regions['shape_attributes']['x']
            y = item_regions['shape_attributes']['y']
            width = item_regions['shape_attributes']['width']
            height = item_regions['shape_attributes']['height']
            
            it_df = pd.DataFrame({'imagename': [str_filename], 'x': [x], 'width': [width], 'y': [y], 'height': [height], 'label': [label]})
            
            cell_df = cell_df.append(it_df, ignore_index = True)
            
            
        except:
            pass
        
for item_images in list(datastore4['_via_img_metadata'].keys()):
    str_filename = datastore4['_via_img_metadata'][item_images]['filename']
    for item_regions in datastore4['_via_img_metadata'][item_images]['regions']:
            
        try:
            for keys in item_regions['region_attributes']['type']: 
                label = str(keys)
                #label = item_regions['region_attributes']['type']
            
            x = item_regions['shape_attributes']['x']
            y = item_regions['shape_attributes']['y']
            width = item_regions['shape_attributes']['width']
            height = item_regions['shape_attributes']['height']
            
            it_df = pd.DataFrame({'imagename': [str_filename], 'x': [x], 'width': [width], 'y': [y], 'height': [height], 'label': [label]})
            
            cell_df = cell_df.append(it_df, ignore_index = True)
            
            
        except:
            pass
        
for item_images in list(datastore5['_via_img_metadata'].keys()):
    str_filename = datastore5['_via_img_metadata'][item_images]['filename']
    for item_regions in datastore5['_via_img_metadata'][item_images]['regions']:
            
        try:
            for keys in item_regions['region_attributes']['type']: 
                label = str(keys)
                #label = item_regions['region_attributes']['type']
            
            x = item_regions['shape_attributes']['x']
            y = item_regions['shape_attributes']['y']
            width = item_regions['shape_attributes']['width']
            height = item_regions['shape_attributes']['height']
            
            it_df = pd.DataFrame({'imagename': [str_filename], 'x': [x], 'width': [width], 'y': [y], 'height': [height], 'label': [label]})
            
            cell_df = cell_df.append(it_df, ignore_index = True)
            
            
        except:
            pass

for item_images in list(datastore6['_via_img_metadata'].keys()):
    str_filename = datastore6['_via_img_metadata'][item_images]['filename']
    for item_regions in datastore6['_via_img_metadata'][item_images]['regions']:
            
        try:
            for keys in item_regions['region_attributes']['type']: 
                label = str(keys)
                #label = item_regions['region_attributes']['type']
            
            x = item_regions['shape_attributes']['x']
            y = item_regions['shape_attributes']['y']
            width = item_regions['shape_attributes']['width']
            height = item_regions['shape_attributes']['height']
            
            it_df = pd.DataFrame({'imagename': [str_filename], 'x': [x], 'width': [width], 'y': [y], 'height': [height], 'label': [label]})
            
            cell_df = cell_df.append(it_df, ignore_index = True)
            
            
        except:
            pass

for item_images in list(datastore7['_via_img_metadata'].keys()):
    str_filename = datastore7['_via_img_metadata'][item_images]['filename']
    for item_regions in datastore7['_via_img_metadata'][item_images]['regions']:
            
        try:
            for keys in item_regions['region_attributes']['type']: 
                label = str(keys)
                #label = item_regions['region_attributes']['type']
            
            x = item_regions['shape_attributes']['x']
            y = item_regions['shape_attributes']['y']
            width = item_regions['shape_attributes']['width']
            height = item_regions['shape_attributes']['height']
            
            it_df = pd.DataFrame({'imagename': [str_filename], 'x': [x], 'width': [width], 'y': [y], 'height': [height], 'label': [label]})
            
            cell_df = cell_df.append(it_df, ignore_index = True)
            
            
        except:
            pass
for item_images in list(datastore8['_via_img_metadata'].keys()):
    str_filename = datastore8['_via_img_metadata'][item_images]['filename']
    for item_regions in datastore8['_via_img_metadata'][item_images]['regions']:
            
        try:
            for keys in item_regions['region_attributes']['type']: 
                label = str(keys)
                #label = item_regions['region_attributes']['type']
            
            x = item_regions['shape_attributes']['x']
            y = item_regions['shape_attributes']['y']
            width = item_regions['shape_attributes']['width']
            height = item_regions['shape_attributes']['height']
            
            it_df = pd.DataFrame({'imagename': [str_filename], 'x': [x], 'width': [width], 'y': [y], 'height': [height], 'label': [label]})
            
            cell_df = cell_df.append(it_df, ignore_index = True)
            
            
        except:
            pass

#Einlesen der Bilder für das Training
def crop_image_to_bbox(img_arr, X_meta):
    """Given an image array, crops the image to just the bounding box provided."""
    shape = img_arr.shape
    x_min = int(X_meta['x'])
    x_max = int(X_meta['x'] + X_meta['width'])
    y_min = int(X_meta['y'])
    y_max = int(X_meta['y'] + X_meta['height'])
    
   
    return img_arr[y_min:y_max, x_min:x_max]  

def get_image_arr(img):
    """Returns the image read out from local disk into a numpy array."""
    return np.asarray(Image.open(os.path.join(images_folder, img)))     
filelist_div = []
filelist_dead = []
filelist_round = []
for entry in range(len(cell_df)):
    
    # Issues with Crop number 144
#    if entry == 144:
#        pass
#    if entry == 6511 or 6547:
#        pass
#    else: 
        #print(cell_df.iloc[entry]['imagename'])
    imagename_crop = 'img_' + str(entry) + '_' + cell_df.iloc[entry]['label'] +'.png'
        #print(imagename_crop)
    plt.imsave(os.path.join(main_folder, r'crops\\' + cell_df.iloc[entry]['label'] + '\\' + imagename_crop), crop_image_to_bbox(get_image_arr(cell_df.iloc[entry]['imagename']), cell_df.iloc[entry]), cmap = 'gray')
    if cell_df.iloc[entry]['label'] == 'div':
        filelist_div.append('img_' + str(entry) + '_' + cell_df.iloc[entry]['label'])
    if cell_df.iloc[entry]['label'] == 'round':
        filelist_round.append('img_' + str(entry) + '_' + cell_df.iloc[entry]['label'])
    if cell_df.iloc[entry]['label'] == 'dead':
        filelist_dead.append('img_' + str(entry) + '_' + cell_df.iloc[entry]['label'])
#Vervielfältigung der div Images:
#from PIL import ImageFilter
from PIL import ImageEnhance

directory_div= os.path.join(main_folder, r'crops\div')
directory_round= os.path.join(main_folder, r'crops\round')
directory_dead= os.path.join(main_folder, r'crops\dead')

for imagefile in filelist_round[::2]:
   im=Image.open(os.path.join(directory_round, imagefile + '.png'))
   flip_im = im.transpose(Image.FLIP_LEFT_RIGHT)
   flip_im.save(os.path.join(directory_round, imagefile+'_flip'+'.png'))
   
for imagefile in filelist_dead[::2]:
   im=Image.open(os.path.join(directory_dead, imagefile + '.png'))
   flip_im = im.transpose(Image.FLIP_LEFT_RIGHT)
   flip_im.save(os.path.join(directory_dead, imagefile+'_flip'+'.png'))
   
for imagefile in filelist_div:
   im=Image.open(os.path.join(directory_div, imagefile + '.png'))
   sharper = ImageEnhance.Sharpness(im)
   brighter = ImageEnhance.Brightness(im)
   contrast = ImageEnhance.Contrast(im)
   sharper_im = sharper.enhance(2.5)
   brighter_im = brighter.enhance(1.2)
   darker_im = brighter.enhance(0.8)
   contrast_high_im = contrast.enhance(1.4)
   contrast_low_im = contrast.enhance(0.8)
   flip_im = im.transpose(Image.FLIP_LEFT_RIGHT)
   sharper_flip = ImageEnhance.Sharpness(flip_im)
   brighter_flip = ImageEnhance.Brightness(flip_im)
   contrast_flip = ImageEnhance.Contrast(flip_im)
   sharper_im_flip = sharper_flip.enhance(2.5)
   flip_brighter_im = brighter_flip.enhance(1.2)
   flip_darker_im = brighter_flip.enhance(0.8)
   flip_contrast_high_im = contrast_flip.enhance(1.4)
   flip_contrast_low_im = contrast_flip.enhance(0.8)
#    im_blur=im.filter(ImageFilter.GaussianBlur)
#    im_unsharp=im.filter(ImageFilter.UnsharpMask)
   sharper_im_flip.save(os.path.join(directory_div, imagefile+'_sharper_flip'+'.png'))
   flip_brighter_im.save(os.path.join(directory_div, imagefile+'_brighter_flip'+'.png'))
   flip_darker_im.save(os.path.join(directory_div, imagefile+'_darker_flip'+'.png'))
   flip_contrast_high_im.save(os.path.join(directory_div, imagefile+'_contrast_high_flip'+'.png'))
   flip_contrast_low_im.save(os.path.join(directory_div, imagefile+'_contrast_low_flip'+'.png'))    
   sharper_im.save(os.path.join(directory_div, imagefile+'_sharper'+'.png'))
   brighter_im.save(os.path.join(directory_div, imagefile+'_brighter'+'.png'))
   darker_im.save(os.path.join(directory_div, imagefile+'_darker'+'.png'))
   contrast_high_im.save(os.path.join(directory_div, imagefile+'_contrast_high'+'.png'))
   contrast_low_im.save(os.path.join(directory_div, imagefile+'_contrast_low'+'.png'))     
#Training
X=[]
Z=[]  
IMG_SIZE=150

LIV_DIR=os.path.join(main_folder, r'crops/liv')
DIV_DIR=os.path.join(main_folder, r'crops/div')
DEAD_DIR=os.path.join(main_folder, r'crops/dead')
ROUND_DIR=os.path.join(main_folder, r'crops/round')

def assign_label(img,cell_type):
    return cell_type

def make_train_data(cell_type,DIR):
    for img in tqdm(os.listdir(DIR)):
        label=assign_label(img,cell_type)
        path = os.path.join(DIR,img)
        img = cv2.imread(path,cv2.IMREAD_COLOR)
        img = cv2.resize(img, (IMG_SIZE,IMG_SIZE))
        
        X.append(np.array(img))
        Z.append(str(label))
        
make_train_data('liv',LIV_DIR)
make_train_data('div',DIV_DIR)
make_train_data('round',ROUND_DIR)
make_train_data('dead',DEAD_DIR)

le=LabelEncoder()
Y=le.fit_transform(Z)
le.inverse_transform([0,1,2,3])
Y=to_categorical(Y,4)
X=np.array(X)
X=X/255
x_train,x_test,y_train,y_test=train_test_split(X,Y,test_size=0.25,random_state=42)


#Bestehendes Modell hier laden
model = Sequential()
model = load_model(os.path.join(main_folder,r'weights/CNN_epochs500_batch128.hdf5'))
model.summary()

#Modell trainieren
np.random.seed(42)
rn.seed(42)
tf.random.set_seed(42)

#Modell Architektur definieren

# # modelling starts using a CNN.

model = Sequential()
model.add(Conv2D(filters = 32, kernel_size = (5,5),padding = 'Same',activation ='relu', input_shape = (150,150,3)))
model.add(MaxPooling2D(pool_size=(2,2)))


model.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same',activation ='relu'))
model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))
 

model.add(Conv2D(filters =96, kernel_size = (3,3),padding = 'Same',activation ='relu'))
model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))

model.add(Conv2D(filters = 96, kernel_size = (3,3),padding = 'Same',activation ='relu'))
model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))

model.add(Flatten())
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dense(4, activation = "softmax"))

#Daten generieren
datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        rotation_range=90,  # randomly rotate images in the range (degrees, 0 to 180)
        zoom_range = 0.2, # Randomly zoom image 
        width_shift_range=0.2,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.2,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=True,  # randomly flip images
        vertical_flip=False)  # randomly flip images
        #brightness_range=[0.5, 1.5])
optimizer = Adam(lr=0.001)
model.compile(optimizer= optimizer,loss='categorical_crossentropy',metrics=['accuracy'])
batch_size=128
epochs=500

callback_checkpoint = ModelCheckpoint(os.path.join(main_folder,r'weights/CNN_epochs{0}_batch{1}.hdf5').format(
        epochs, batch_size), monitor='val_loss', verbose=0, save_best_only=True, save_weights_only=False, mode='min')
#y_train_labels = cell_df['label']
#from sklearn.utils import class_weight
#class_weight = class_weight.compute_class_weight('balanced', np.unique(y_train_labels), y_train_labels)
callback_history = History()
Training_history = model.fit_generator(datagen.flow(x_train,y_train, batch_size=batch_size),
                              epochs = epochs, validation_data = (x_test,y_test),
                              verbose = 1, steps_per_epoch=x_train.shape[0] // batch_size, callbacks=[callback_history, callback_checkpoint])#, class_weight = class_weight)

#Accuracy und loss des Modells
plt.figure(1)
plt.plot(Training_history.history['loss'])
plt.plot(Training_history.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epochs')
plt.legend(['train', 'test'])
plt.show()

plt.figure(2)
plt.plot(Training_history.history['accuracy'])
plt.plot(Training_history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epochs')
plt.legend(['train', 'test'])
plt.show()

#Modellperformance am Validierungsset
# getting predictions on val set.
pred=model.predict(x_test)
pred_digits=np.argmax(pred,axis=1)

# Confusion matrix
target_names = le.inverse_transform([0,1,2,3])
cm = confusion_matrix(np.argmax(y_test, axis=1), pred_digits)

plt.figure(3)
ax = sns.heatmap(cm, cmap = 'Blues', annot = True, fmt = '.0f', xticklabels = target_names, yticklabels = target_names)
ax.set(xlabel='Predicted Class', ylabel='True Class')
plt.show()

print(classification_report(np.argmax(y_test, axis=1), pred_digits, target_names=target_names))

# now storing some properly as well as misclassified indexes'.

#prop_class=[]
#mis_class=[]
#
#i=0
#
#for i in range(len(y_test)):
#    if(np.argmax(y_test[i])==pred_digits[i]):
#        prop_class.append(i)
#    if(len(prop_class)==8):
#        break
#
#i=0
#for i in range(len(y_test)):
#    if (not np.argmax(y_test[i])==pred_digits[i]):
#        mis_class.append(i)
#    if(len(mis_class)==8):
#        break
#
## Ignore  the warnings
#import warnings
#warnings.filterwarnings('always')
#warnings.filterwarnings('ignore')
