from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D, BatchNormalization
from keras.optimizers import Adam 

model = Sequential()

model.add(Convolution2D(32,3,3,init='he_normal',border_mode='same',input_shape=(3,32,32)))
model.add(BatchNormalization(mode=0,axis=1))
model.add(Activation('relu'))
model.add(Convolution2D(32,3,3,init='he_normal',border_mode='same')
model.add(BatchNormalization(mode=0,axis=1))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))


model.add(Convolution2D(32,3,3,init='he_normal',border_mode='same',input_shape=(3,32,32)))
model.add(BatchNormalization(mode=0,axis=1))
model.add(Activation('relu'))
model.add(Convolution2D(32,3,3,init='he_normal',border_mode='same')
model.add(BatchNormalization(mode=0,axis=1))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Flatten())
model.add(Dense(256))
model.add(BatchNormalization(mode=1))
model.add(Activation('relu'))

model.add(Dense(10))
model.add(Activation('softmax'))

adam = Adam(lr=1e-3)
model.compile(loss='categorical_crossentropy', optimizer=adam)

model.fit(X_train, Y_train, batch_size=200, nb_epoch=1)
