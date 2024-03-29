#!/usr/bin/env python
# coding: utf-8

# In[8]:


pip install tensorflow


# In[53]:


import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import pathlib
from tensorflow import keras
import tensorflow.keras 
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
import tkinter as tk
from tkinter import filedialog


# In[54]:


data_dir = "D:\machine learning code\\face mask detection\Face Mask Dataset\Train"
data_dir = pathlib.Path(data_dir)


# In[55]:


batch_size = 16
img_height = 64
img_width = 64


# In[56]:


# Reading Training images from the directory
train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    data_dir, validation_split = 0.2, subset = "training", seed = 123,
    image_size = (img_height, img_width), batch_size = batch_size)


# In[57]:


# Reading validation images from the directory
val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    data_dir, validation_split = 0.2, subset = "validation", seed = 123,
    image_size = (img_height, img_width), batch_size = batch_size)


# In[58]:


class_names = train_ds.class_names
print(class_names)


# In[59]:


#Memory optimization and speed up execution
AUTOTUNE = tf.data.experimental.AUTOTUNE
train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size = AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size = AUTOTUNE)


# In[60]:


num_class = 2


# In[61]:


#Definning CNN
model = Sequential([
    layers.experimental.preprocessing.Rescaling(1./255, input_shape = (img_height, img_width, 3)),
    layers.Conv2D(16,(3,3), padding = 'same', activation = 'relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(32,(3,3), padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(64, (3,3), padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Dropout(0.2),
    layers.Flatten(),
    layers.Dense(64, activation = 'relu'),
    layers.Dense(num_class)
])

noepochs = 7


# In[62]:


model.compile(optimizer = 'adam', loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits = True),
                  metrics = ['accuracy'])
#training the model
mymodel = model.fit(train_ds, validation_data = val_ds, epochs = noepochs)


# In[63]:


acc = mymodel.history['accuracy']
val_acc = mymodel.history['val_accuracy']
loss = mymodel.history['loss']
val_loss = mymodel.history['val_loss']
epochs_range = range(noepochs)


# In[64]:


plt.figure(figsize=(15, 15)) #creates figure for the plot
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label = 'Training Accuracy')
plt.plot(epochs_range, val_acc, label = 'Validation Accuracy')
plt.legend(loc = 'upper center')
plt.title('Training and validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label = 'Training loss')
plt.plot(epochs_range, val_loss, label = 'Validation loss')
plt.legend(loc = 'upper center')
plt.title('Training and validation loss')
plt.show()


# In[65]:


# Function to test single image
def recogout():
    root = tk.Tk()
    root.withdraw()
    img_path = filedialog.askopenfilename()
    img = keras.preprocessing.image.load_img(img_path, target_size = (img_height, img_width))
    img_array = keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)
    predictions = model.predict(img_array)
    score = tf.nn.softmax(predictions[0])

    print("This image most likely belongs to {} with a {:.2f} percent confidence."
          .format(class_names[np.argmax(score)], 100*np.max(score)))


# In[66]:


from tkinter import *
import tkinter as tk
from tkinter import filedialog


# In[71]:


recogout()


# In[ ]:




