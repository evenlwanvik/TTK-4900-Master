from keras import layers
from keras.layers import Input, Add, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D, AveragePooling2D, MaxPooling2D, GlobalMaxPooling2D, Dropout
from keras.models import Model, Sequential  
from keras.utils import plot_model
from keras.layers.merge import concatenate
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
import keras.backend as K
K.set_image_data_format('channels_last')

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # Turn off tensorflow debugging logs
#K.set_learning_phase(1)

##################### MODELS #####################

def VGG16(input_shape, classes):
    model = Sequential()
    model.add(Conv2D(input_shape=input_shape,filters=64,kernel_size=(3,3),padding="same", activation="relu"))
    model.add(Conv2D(filters=64,kernel_size=(3,3),padding="same", activation="relu"))
    model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2)))
    model.add(Dropout(0.20))
    model.add(Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu"))
    model.add(Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu"))
    model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2)))
    model.add(Dropout(0.20))
    model.add(Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))
    model.add(Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))
    model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2)))
    model.add(Dropout(0.20))
    model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
    model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
    model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2)))
    model.add(Dropout(0.20))
    model.add(Flatten())
    model.add(Dense(units=256,activation="relu"))
    model.add(Dense(units=128,activation="relu"))
    model.add(Dense(units=classes, activation="softmax"))
    return model


# A simple network used for the mnist dataset
def mnist(input_shape, classes):
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3),activation='relu',kernel_initializer='he_normal',input_shape=input_shape))
    model.add(Conv2D(32, kernel_size=(3, 3),activation='relu',kernel_initializer='he_normal'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.20))
    model.add(Conv2D(64, (3, 3), activation='relu',padding='same',kernel_initializer='he_normal'))
    model.add(Conv2D(64, (3, 3), activation='relu',padding='same',kernel_initializer='he_normal'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))
    model.add(Conv2D(128, (3, 3), activation='relu',padding='same',kernel_initializer='he_normal'))
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(classes, activation='softmax'))
    #model.compile(optimizer='adagrad', loss='binary_crossentropy', metrics=['accuracy'])
    return model


def conv2d(x,numfilt,filtsz,pad='same',act=True,name=None):
  x = Conv2D(numfilt,filtsz,padding=pad,data_format='channels_last',use_bias=False,name=name+'conv2d')(x)
  x = BatchNormalization(axis=3,scale=False,name=name+'conv2d'+'bn')(x)
  if act:
    x = Activation('relu',name=name+'conv2d'+'act')(x)
  return x

def inception_resnet_A(x,name=None):
    pad = 'same'
    branch0 = conv2d(x,32,1,pad,True,name=name+'b0')
    branch1 = conv2d(x,32,1,pad,True,name=name+'b1_1')
    branch1 = conv2d(branch1,32,3,pad,True,name=name+'b1_2')
    branch2 = conv2d(x,32,1,pad,True,name=name+'b2_1')
    branch2 = conv2d(branch2,48,3,pad,True,name=name+'b2_2')
    branch2 = conv2d(branch2,64,3,pad,True,name=name+'b2_3')
    branches = concatenate([branch0,branch1,branch2], axis=-1)
    # concatenate filters, assumes filters/channels last
    filt_exp_1x1 = conv2d(branches,384,1,pad,False,name=name+'filt_exp_1x1')
    return concatenate([x, filt_exp_1x1], axis=-1)

def inception_resnet_B(x,name=None):
    pad = 'same'
    branch0 = conv2d(x,96,1,pad,True,name=name+'b0')
    branch1 = conv2d(x,80,1,pad,True,name=name+'b1_1')
    branch1 = conv2d(branch1,120,[1,7],pad,True,name=name+'b1_2')
    branch1 = conv2d(branch1,140,[7,1],pad,True,name=name+'b1_3')
    branches = concatenate([branch0,branch1], axis=-1)
    filt_exp_1x1 = conv2d(branches,256,1,pad,False,name=name+'filt_exp_1x1')
    return concatenate([x, filt_exp_1x1], axis=-1)

def inception_resnet_C(x,name=None):
    pad = 'same'
    branch0 = conv2d(x,120,1,pad,True,name=name+'b0')
    branch1 = conv2d(x,120,1,pad,True,name=name+'b1_1')
    branch1 = conv2d(branch1,160,[1,3],pad,True,name=name+'b1_2')
    branch1 = conv2d(branch1,160,[3,1],pad,True,name=name+'b1_3')
    branches = concatenate([branch0,branch1], axis=-1)
    filt_exp_1x1 = conv2d(branches,384,1,pad,False,name=name+'filt_exp_1x1')
    return concatenate([x, filt_exp_1x1], axis=-1)

def inception_resnet_v2(input_shape, classes, model_fpath):
    data_in = Input(shape=input_shape)   
    x = conv2d(data_in,32,3,'same',True,name='conv1')
    x = conv2d(x,64,3,'same',True,name='conv2')
    x_11 = AveragePooling2D(2,strides=1,padding='same',name='avgpool_1')(x)
    x_12 = conv2d(x,64,3,'same',True,name='stem_br_12')
    x = concatenate([x_11,x_12], axis=-1, name = 'stem_concat_1')
    #x = AveragePooling2D(2,strides=1,padding='same',name='avgpool_2')(x)
    x = Dropout(0.2)(x)

    x = inception_resnet_A(x,name='moduleA1')
    x = AveragePooling2D(2,strides=1,padding='same',name='avgpool_3')(x)
    x = Dropout(0.2)(x)

    #x = inception_resnet_A(x,name='moduleA2')
    #x = AveragePooling2D(2,padding='same',name='avgpool_4')(x)
    #x = Dropout(0.2)(x)

    x = Flatten()(x)
    #x = Dense(64, activation='relu')(x)
    #x = Dropout(0.2)(x)

    x = Dense(classes, activation='softmax')(x)

    checkpoint = ModelCheckpoint(model_fpath, monitor='val_acc', 
                        verbose=1, save_best_only=True, mode='max')

    early = EarlyStopping(monitor="val_loss", 
                      mode="min", 
                      patience=4, restore_best_weights=True)

    callbacks_list = [checkpoint, early]

    model = Model(data_in,x,name='custom_inception_resnet_v2')

    model.summary()
    
    return model, callbacks_list

def my_model_inception(input_shape, classes):
    # (no of inputs + no of outputs)^0.5 + (1 to 10)
    # ~ sqrt(1000) = 100

    visible = Input(shape=input_shape)
    # add inception block 1
    layer = inception_resnet_A(visible, name='moduleA1')
    layer = AveragePooling2D((2, 2))(layer)
    layer = Dropout(0.25)(layer)
    layer = inception_resnet_A(visible, name='moduleA2')
    layer = AveragePooling2D((2, 2))(layer)
    layer = Dropout(0.25)(layer)
    #layer = BatchNormalization()(layer)
    #layer = Conv2D(32, (3, 3), activation='relu',padding='same',kernel_initializer='he_normal')(layer)
    #layer = Conv2D(64, (3, 3), activation='relu',padding='same',kernel_initializer='he_normal')(layer)
    #layer = AveragePooling2D((2, 2))(layer)
    #layer = Dropout(0.25)(layer)
    # add inception block 1
    #layer = inception_module_A(layer, 32, 32, 64, 16, 32, 16)
    layer = inception_resnet_A(layer, name='moduleA3')
    layer = AveragePooling2D(pool_size=(2, 2), padding='valid')(layer)
    layer = Dropout(0.25)(layer)
    #layer = BatchNormalization()(layer)
    layer = Flatten()(layer)
    layer = Dense(512, activation='relu')(layer)
    layer = Dense(128, activation='relu')(layer)
    layer = Dropout(0.25)(layer)
    layer = Dense(classes, activation='softmax')(layer)
    model = Model(inputs=visible, outputs=layer)
    # summarize model
    model.summary()

    return model


def my_model(input_shape, classes):
    model = Sequential()
    model.add(Conv2D(64, kernel_size=(3,3), padding='same',activation='relu',kernel_initializer='he_normal',input_shape=input_shape))
    model.add(Conv2D(32, kernel_size=(3,3), padding='same',activation='relu',kernel_initializer='he_normal'))
    model.add(AveragePooling2D((2, 2), padding='same'))
    model.add(Conv2D(64, kernel_size=(5,5), padding='same', activation='relu'))
    model.add(Conv2D(128, kernel_size=(3,3), padding='same', activation='relu'))
    model.add(BatchNormalization(axis=-1))
    model.add(AveragePooling2D(pool_size=(2,2), padding='same'))
    model.add(Dropout(0.25))
    model.add(Conv2D(64, kernel_size=(3,3), padding='same', activation='relu'))
    #model.add(BatchNormalization(axis=-1))
    model.add(AveragePooling2D(pool_size=(2,2), padding='same'))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.25))
    model.add(Dense(classes, activation='softmax'))
    return model


def best_sofar(input_shape, classes):
    model = Sequential()
    model.add(Conv2D(64, kernel_size=(3,3), padding='same',activation='relu',kernel_initializer='he_normal',input_shape=input_shape))
    model.add(Conv2D(32, kernel_size=(3,3), padding='same',activation='relu',kernel_initializer='he_normal'))
    model.add(MaxPooling2D((2, 2), padding='same'))
    model.add(Dropout(0.25))  
    model.add(Conv2D(64, kernel_size=(3,3), padding='same',activation='relu',kernel_initializer='he_normal'))
    model.add(Conv2D(32, kernel_size=(3,3), padding='same',activation='relu',kernel_initializer='he_normal'))  
    model.add(MaxPooling2D((2, 2), padding='same'))
    model.add(Dropout(0.25)) 
    model.add(Conv2D(64, kernel_size=(3,3), padding='same',activation='relu',kernel_initializer='he_normal'))  
    model.add(Conv2D(32, kernel_size=(3,3), padding='same',activation='relu',kernel_initializer='he_normal'))
    model.add(MaxPooling2D((2, 2), padding='same'))
    model.add(Dropout(0.25)) 
    #model.add(Conv2D(64, kernel_size=(3,3), padding='same',activation='relu',kernel_initializer='he_normal'))
    #model.add(Conv2D(32, kernel_size=(3,3), padding='same',activation='relu',kernel_initializer='he_normal'))  
    #model.add(AveragePooling2D((2, 2), padding='same'))
    #model.add(Dropout(0.25)) 
    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dense(256, activation='relu'))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(classes, activation='softmax'))
    model.summary()
    return model


def test_shallow(input_shape, classes):
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(5,5), padding='same',activation='relu',kernel_initializer='he_normal',input_shape=input_shape))
    model.add(Conv2D(32, kernel_size=(5,5), padding='same',activation='relu',kernel_initializer='he_normal'))
    model.add(MaxPooling2D((2, 2), padding='same'))
    model.add(Dropout(0.25)) 
    model.add(Conv2D(64, kernel_size=(3,3), padding='same',activation='relu',kernel_initializer='he_normal'))  
    model.add(Conv2D(32, kernel_size=(3,3), padding='same',activation='relu',kernel_initializer='he_normal'))
    model.add(MaxPooling2D((2, 2), padding='same'))
    model.add(Dropout(0.25)) 
    model.add(Conv2D(64, kernel_size=(3,3), padding='same',activation='relu',kernel_initializer='he_normal'))  
    model.add(Conv2D(64, kernel_size=(3,3), padding='same',activation='relu',kernel_initializer='he_normal'))
    model.add(MaxPooling2D((2, 2), padding='same'))
    model.add(Dropout(0.25)) 
    #model.add(Conv2D(64, kernel_size=(3,3), padding='same',activation='relu',kernel_initializer='he_normal'))
    #model.add(Conv2D(32, kernel_size=(3,3), padding='same',activation='relu',kernel_initializer='he_normal'))  
    #model.add(AveragePooling2D((2, 2), padding='same'))
    #model.add(Dropout(0.25)) 
    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(classes, activation='softmax'))
    model.summary()
    return model

def simple_model_for_visualization(input_shape, classes):

    model = Sequential()

    model.add(Conv2D(32, (3, 3), padding='same', input_shape = input_shape, activation = 'relu'))
    model.add(Conv2D(32, (3, 3), activation='relu'))
    #model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(AveragePooling2D((2, 2), padding='same'))
    model.add(Dropout(0.25)) # test 0.5
    # Adding a second convolutional layer
    model.add(Conv2D(64, (3, 3), padding='same', activation = 'relu'))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(AveragePooling2D((2, 2), padding='same'))
    model.add(Dropout(0.25)) 
    # Adding a third convolutional layer
    model.add(Conv2D(64, (3, 3), padding='same', activation = 'relu'))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(AveragePooling2D((2, 2), padding='same'))
    model.add(Dropout(0.25)) 
    # Step 3 - Flattening
    model.add(Flatten())
    # Step 4 - Full connection
    model.add(Dense(units = 512, activation = 'relu'))
    model.add(Dense(units = 256, activation = 'relu'))
    model.add(Dropout(0.25)) 
    model.add(Dense(units = classes, activation = 'softmax'))
    model.summary()
    return model


def conv2d_2(x, filter_size, kernelsize, padding, name=None):
    x = Conv2D(filter_size, kernelsize, padding=padding,kernel_initializer='he_normal', name=name+'conv2d')(x)
    #x = BatchNormalization(axis=-1, name=name+'conv2d'+'bn')(x)
    x = Activation('relu', name=name+'conv2d'+'act')(x)
    return x


def best_sofar_resnet(input_shape, classes):

    data_in = Input(shape=input_shape)   
    # VGG block 1 
    x_in = data_in
    x = conv2d_2(x_in,64,3,'same',name='block_11')
    x = conv2d_2(x_in,32,3,'same',name='block_12')
    x = concatenate([x_in,x], axis=-1, name = 'resnet_1_conc')
    x = AveragePooling2D((2, 2), padding='same')(x)
    x = Dropout(0.25)(x)  
    # VGG block 2
    x_in = x
    x = conv2d_2(x,64,3,'same',name='block_21')
    x = conv2d_2(x,32,3,'same',name='block_22')
    x = concatenate([x_in,x], axis=-1, name = 'resnet_2_conc')
    x = AveragePooling2D((2, 2), padding='same')(x)
    x = Dropout(0.25)(x)
    # VGG block 3
    x_in = x
    x = conv2d_2(x,64,3,'same',name='block_3')
    x = concatenate([x_in,x], axis=-1, name = 'resnet_3_conc')
    x = AveragePooling2D((2, 2), padding='same')(x)
    x = Dropout(0.25)(x)
    # Final
    x = Flatten()(x)
    x = Dense(516, activation='relu')(x)
    x = Dense(128, activation='relu')(x)
    x = Dense(classes, activation='softmax')(x)
    # Model
    model = Model(inputs=data_in, outputs=x)
    model.summary()
    #plot_model(model, to_file='ResNet.png')
    return model


def resnet(x,name=None):
    pad = 'same'
    branch0 = conv2d(x,32,1,pad,True,name=name+'b0')
    branch1 = conv2d(x,32,1,pad,True,name=name+'b1_1')
    branch2 = conv2d(branch2,64,3,pad,True,name=name+'b2_3')
    branches = concatenate([branch0,branch1,branch2], axis=-1)
    # concatenate filters, assumes filters/channels last
    filt_exp_1x1 = conv2d(branches,384,1,pad,False,name=name+'filt_exp_1x1')
    return concatenate([x, filt_exp_1x1], axis=-1)


'''
def shallow_resnet(input_shape, classes):

    data_in = Input(shape=input_shape) 

    layer = model.add(Conv2D(64, kernel_size=(3,3), padding='same',activation='relu',kernel_initializer='he_normal',input_shape=input_shape))((2, 2))(layer)



    model = Sequential()
    
    model.add(Conv2D(32, kernel_size=(3,3), padding='same',activation='relu',kernel_initializer='he_normal'))
    model.add(AveragePooling2D((2, 2), padding='same'))
    model.add(Dropout(0.25))  
    model.add(Conv2D(64, kernel_size=(3,3), padding='same',activation='relu',kernel_initializer='he_normal',input_shape=input_shape))
    model.add(Conv2D(32, kernel_size=(3,3), padding='same',activation='relu',kernel_initializer='he_normal'))  
    model.add(AveragePooling2D((2, 2), padding='same'))
    model.add(Dropout(0.25)) 
    model.add(Conv2D(32, kernel_size=(3,3), padding='same',activation='relu',kernel_initializer='he_normal'))  
    model.add(AveragePooling2D((2, 2), padding='same'))
    model.add(Dropout(0.25)) 
    model.add(Conv2D(32, kernel_size=(3,3), padding='same',activation='relu',kernel_initializer='he_normal'))  
    model.add(AveragePooling2D((2, 2), padding='same'))
    model.add(Dropout(0.25)) 
    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(classes, activation='softmax'))
    model.summary()
    #plot_model(model, to_file='ResNet.png')
    return model
'''