import keras
from keras.applications import VGG16
from keras.models import Model
from keras.layers import Dense, Flatten
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import seaborn as sns

# parameters for data loading and image preprocessing
num_classes = 14
img_width, img_height = 200, 149
batch_size = 32
epochs=100
train_dir = 'path/to/dataset'


base_model = VGG16(weights='imagenet', include_top=False, input_shape=(img_height, img_width, 3))
x = base_model.output
x = Flatten()(x)
x = Dense(256, activation='relu')(x)
predictions = Dense(num_classes, activation='softmax')(x)
model = Model(inputs=base_model.input, outputs=predictions)

# freeze all layers in the base model except the last two
for layer in base_model.layers[:-2]:
    layer.trainable = False

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

train_datagen = ImageDataGenerator(
    rescale=1./255, 
    shear_range=0.2, 
    zoom_range=0.2, 
    horizontal_flip=True,
    validation_split=0.2)


test_datagen = ImageDataGenerator(rescale=1./255)
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(img_height, img_width), 
    batch_size=batch_size, 
    class_mode='categorical',
    subset='training')

test_generator = test_datagen.flow_from_directory(
    train_dir, 
    target_size=(img_height, img_width), 
    batch_size=batch_size, 
    class_mode='categorical',
    subset='validation')

history = model.fit(train_generator, epochs=epochs, validation_data=test_generator)

model.save('vgg16_model.h5')

# Plotting the training and validation loss and accuracy
sns.set_style("darkgrid")
plt.figure(figsize=(8, 6))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Training and Validation Metrics')
plt.ylabel('Metric Value')
plt.xlabel('Epoch')
plt.legend()
plt.savefig('metric.png')
