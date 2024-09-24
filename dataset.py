import numpy
import tensorflow as tf

train_directory = "C:/Users/yud/PycharmProjects/pythonProject1/train"
val_directory = "C:/Users/yud/PycharmProjects/pythonProject1/test"
class_labels  = ['angry', 'disgusted', 'fearful', 'happy', 'neutral', 'sad', 'surprised']
class_labels_emojis = ["👿", "🤢" , "😱" , "😊" , "😐 ", "😔" , "😲" ]

#import PIL
#PIL.Image.open("C:/Users/yud/PycharmProjects/pythonProject1/train/angry/im0.png").show()

import matplotlib.pyplot as plt
import cv2
import matplotlib
matplotlib.use('TkAgg')

path=[]
for i in range(len(class_labels)):
    path.append("C:/Users/yud/PycharmProjects/pythonProject1/train/"+class_labels[i]+"/im0.png")
print(path)
print(path[0])

image1=cv2.imread(path[0])
image2=cv2.imread(path[1])
image3=cv2.imread(path[2])
image4=cv2.imread(path[3])
image5=cv2.imread(path[4])
image6=cv2.imread(path[5])
image7=cv2.imread(path[6])

fig = plt.figure()
fig.add_subplot(2, 4, 1)
plt.imshow(image1)
plt.axis('off')
plt.title(class_labels[0])

fig.add_subplot(2, 4, 2)
plt.imshow(image2)
plt.axis('off')
plt.title(class_labels[1])

fig.add_subplot(2, 4, 3)
plt.imshow(image3)
plt.axis('off')
plt.title(class_labels[2])

fig.add_subplot(2, 4, 4)
plt.imshow(image4)
plt.axis('off')
plt.title(class_labels[3])

fig.add_subplot(2, 4, 5)
plt.imshow(image5)
plt.axis('off')
plt.title(class_labels[4])

fig.add_subplot(2, 4, 6)
plt.imshow(image6)
plt.axis('off')
plt.title(class_labels[5])

fig.add_subplot(2, 4, 7)
plt.imshow(image7)
plt.axis('off')
plt.title(class_labels[6])

import os
listy=[]
for i in range(7):
    _, _, files = next(os.walk("C:/Users/yud/PycharmProjects/pythonProject1/train/"+class_labels[i]))
    file_count = len(files)
    listy.append(file_count)
print(listy)

import plotly.express as px
fig = px.bar(x = class_labels_emojis,
             y = listy ,
             color = numpy.unique(class_labels),
             color_continuous_scale="Emrld")
fig.update_xaxes(title="Emotions")
fig.update_yaxes(title = "Number of Images")
fig.update_layout(showlegend = True,
    title = {
        'text': 'Train Data Distribution ',
        'y':0.95,
        'x':0.5,
        'xanchor': 'center',
        'yanchor': 'top'})
fig.show()

#############Dataset distribution####################################################
train_dataset = tf.keras.utils.image_dataset_from_directory(
    train_directory,
    labels='inferred',
    label_mode='categorical',
    class_names= class_labels,
    color_mode='rgb',
    batch_size=32,
    image_size=(48, 48),
    shuffle=True,
    seed=99,
)


val_dataset = tf.keras.utils.image_dataset_from_directory(
    val_directory,
    labels='inferred',
    label_mode='categorical',
    class_names= class_labels,
    color_mode='rgb',
    batch_size=1,
    image_size=(48,48),
    shuffle=True,
    seed=99,
)

import numpy as np
import tensorflow as tf
import cv2

data = tf.keras.utils.image_dataset_from_directory('C:/Users/yud/PycharmProjects/testDataset/train',batch_size=1,image_size=(171,256))
for batch_x, batch_y in data:
    for i, x in enumerate(batch_x):
       x = np.asarray(x)
       cv2.imshow("",x)

data = tf.keras.utils.image_dataset_from_directory('img',batch_size=1,image_size=(171,256))
for batch_x, batch_y in data:
    for i, x in enumerate(batch_x):
       x = np.asarray(x)
       cv2_imshow(x)

import matplotlib.pyplot as plt

def display_one_image(image, title, subplot, color):
    plt.subplot(subplot)
    plt.axis('off')
    plt.imshow(image)
    plt.title(title, fontsize=16)

def display_nine_images(images, titles, title_colors=None):
    subplot = 331
    plt.figure(figsize=(13,13))
    for i in range(9):
        color = 'black' if title_colors is None else title_colors[i]
        display_one_image(images[i], titles[i], 331+i, color)
    plt.tight_layout()
    plt.subplots_adjust(wspace=0.1, hspace=0.1)
    plt.show()

sample = tf.data.Dataset.from_tensor_slices(list(train_dataset))
print(sample[0])

images, classes = next(img_datagen)
class_idxs = np.argmax(classes, axis=-1)
labels = [class_labels[idx] for idx in class_idxs]
display_nine_images(images, labels)

fig = px.bar(x = class_labels_emojis,
             y = [list(training_dataset.class_labels).count(i) for i in numpy.unique(training_dataset.class_labels)] ,
             color = np.unique(training_dataset.class_labels) ,
             color_continuous_scale="Emrld")
fig.update_xaxes(title="Emotions")
fig.update_yaxes(title = "Number of Images")
fig.update_layout(showlegend = True,
    title = {
        'text': 'Train Data Distribution ',
        'y':0.95,
        'x':0.5,
        'xanchor': 'center',
        'yanchor': 'top'})
fig.show()