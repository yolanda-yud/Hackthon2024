from tensorflow.keras.models import Model
from tensorflow.keras.layers import Layer
from tensorflow.keras.layers import (GlobalAveragePooling2D, Activation, MaxPooling2D, Add, Conv2D, MaxPool2D, Dense,
                                     Flatten, InputLayer, BatchNormalization, Input, Embedding, Permute,
                                     Dropout, RandomFlip, RandomRotation, LayerNormalization, MultiHeadAttention,
                                     RandomContrast, Rescaling, Resizing, Reshape)
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import sklearn ### machine learning library
from sklearn.metrics import confusion_matrix, roc_curve### metrics


from tensorflow.keras.losses import BinaryCrossentropy,CategoricalCrossentropy, SparseCategoricalCrossentropy
from tensorflow.keras.metrics import Accuracy,TopKCategoricalAccuracy, CategoricalAccuracy, SparseCategoricalAccuracy
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers  import L2, L1

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

training_dataset = (
    train_dataset
    .prefetch(tf.data.AUTOTUNE)
)

validation_dataset = (
    val_dataset
    .prefetch(tf.data.AUTOTUNE)
)

base_model = tf.keras.applications.VGG16(weights='imagenet', include_top=False, input_shape=(48, 48, 3))

# Add custom layers on top of VGG16
x = base_model.output
x = GlobalAveragePooling2D()(x)  # Reduce spatial dimensions
x = Dense(1024, activation='relu')(x)  # Add a fully connected layer
predictions = Dense(7, activation='softmax')(x)  # Final output layer for 7 classes

# Create the model
model = Model(inputs=base_model.input, outputs=predictions)

print(model.summary())

loss_function = tf.keras.losses.CategoricalCrossentropy()
metrics = [tf.keras.metrics.CategoricalAccuracy(name = "accuracy"), tf.keras.metrics.TopKCategoricalAccuracy(k=4, name = "top_k_accuracy")]
model.compile(
  optimizer = Adam(learning_rate = 1e-3),
  loss = loss_function,
  metrics = metrics,
)

history = model.fit(
  training_dataset,
  validation_data = validation_dataset,
  batch_size= 32,
  epochs = 3,#20
  verbose = 1
)

#########model metrics##################################################################
import warnings
warnings.filterwarnings('ignore')
print(tf.__version__)

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train_loss', 'val_loss'])
plt.show()

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train_accuracy', 'val_accuracy'])
plt.show()

model.save('face_detection_model.h5')

##########label prediction#############################################################
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
#angry example
#input_img_path='C:/Users/yud/PycharmProjects/pythonProject1/test/angry/im45.png'
#happy example
#input_img_path='C:/Users/yud/PycharmProjects/pythonProject1/test/happy/im2.png'
#sad example
input_img_path='C:/Users/yud/PycharmProjects/pythonProject1/test/sad/im20.png'

input_img=cv2.imread(input_img_path)
plt.imshow(input_img)
image = load_img(input_img_path, target_size=(48, 48))
image = img_to_array(image)
# reshape data for the model
image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
pred = model.predict(image)
class_labels  = ['angry', 'disgusted', 'fearful', 'happy', 'neutral', 'sad', 'surprised']
# retrieve the most likely result, e.g. highest probability
# print the classification
#print(pred[0])
dict1={}
for i in range(len(class_labels)-1):
    if i not in dict1:
        dict1[class_labels[i]]=pred[0][i]
print(dict1)
for k,v in dict1.items():
    if v==max(pred[0]):
        print(k)

