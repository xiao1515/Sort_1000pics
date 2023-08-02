# -*- coding: utf-8 -*-
"""
Created on Wed Aug  2 14:20:09 2023

@author: Joe
"""
from keras.models import Model
from keras.layers import Flatten
from keras.layers import Dense,Dropout
from keras.layers import Input
from keras.layers import Conv2D
from keras.layers import MaxPooling2D,BatchNormalization


def VGG16(input_shape,classes):
 
    img_input = Input(shape=input_shape)

    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1')(img_input)
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)
    x = BatchNormalization()(x)

    # Block 2
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block2_conv1')(x)
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block2_conv2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)
    x = BatchNormalization()(x)
    # Block 3
    
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block3_conv1')(x)
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block3_conv2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)
    x = BatchNormalization()(x)
    # Block 4
    
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block4_conv1')(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block4_conv2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)
    x = BatchNormalization()(x)
    # Block 5
    
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block5_conv1')(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block5_conv2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(x)
    x = BatchNormalization()(x)
    
    # Classification block
    x = Flatten(name='flatten')(x)
    x = Dense(512, activation='relu', name='fc1')(x)
    x = Dropout(0.2)(x)
    x = Dense(256, activation='relu', name='fc2')(x)
    x = Dropout(0.5)(x)
    x = Dense(10, activation='softmax', name='predictions')(x)

    inputs = img_input
    # Create model.
    model = Model(inputs, x, name='vgg16')

   
    return model


#------------------------------------------ResNet50-------------------------------------
def build_model(preModel, num_classes, input_data, Adam):
    pred_model = preModel(include_top=False, weights='imagenet',
                          input_shape=input_data,
                          pooling='max', classifier_activation='softmax')
    output_layer = Dense(
        num_classes, activation="softmax", name="output_layer")

    model = Model(
        pred_model.inputs, output_layer(pred_model.output))

    model.compile(optimizer=Adam,
                  loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model