#!/usr/bin/env python
# coding: utf-8

# In[18]:


from keras import models
from keras.layers import Dense, Dropout
from keras.utils import to_categorical
from keras.datasets import mnist
from keras.utils.vis_utils import model_to_dot
from IPython.display import SVG
import livelossplot
import pandas as pd
import numpy as np


# In[62]:


NUM_ROWS = 500
NUM_COLS = 500
NUM_CLASSES = 10
BATCH_SIZE = 30
EPOCHS = 10


# In[72]:


#Data Preprocessing  
plot_losses = livelossplot.PlotLossesKeras()

x_train = pd.read_csv('madelon/madelon_train.data', delimiter=" ", header=None)
x_test = pd.read_csv('madelon/madelon_valid.data', delimiter=" ", header=None)
y_train = pd.read_csv('madelon/madelon_train.labels', delimiter=" ", header=None)
y_test = pd.read_csv('madelon/madelon_valid.labels', delimiter=" ", header=None)

x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255

x_test = x_test.drop(x_test.columns[500], axis=1)
x_train = x_train.drop(x_train.columns[500], axis=1)

y_train = to_categorical(y_train, NUM_CLASSES)
y_test = to_categorical(y_test, NUM_CLASSES)


# In[73]:


def data_summary(x_train, y_train, x_test, y_test):
    """Summarize current state of dataset"""
    print('Train images shape:', x_train.shape)
    print('Train labels shape:', y_train.shape)
    print('Test images shape:', x_test.shape)
    print('Test labels shape:', y_test.shape)
    print('Train labels:', y_train)
    print('Test labels:', y_test)


# In[74]:


data_summary(x_train, y_train, x_test, y_test)


# In[75]:


model = models.Sequential()
model.add(Dense(512, activation='relu', input_shape=(500, )))
model.add(Dropout(0.5))
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(10, activation='softmax'))


# In[76]:


# Compile model
model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Train model
model.fit(x_train, y_train,
          batch_size=BATCH_SIZE,
          epochs=EPOCHS,
          callbacks=[plot_losses],
          verbose=1,
          validation_data=(x_test, y_test))

score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])


# In[42]:


model.summary()


# In[ ]:


# Output network visualization
SVG(model_to_dot(model).create(prog='dot', format='svg'))

