import keras
from keras.applications import VGG16
from keras.models import Model
from keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.models import load_model, Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator,load_img, img_to_array
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# parameters for data loading and image preprocessing
num_classes = 14
img_width, img_height = 200, 149
batch_size = 64
epochs=100
train_dir = 'path/to/dataset'
model_saveas = 'model.h5'

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

val_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation')

def train_model():
    vgg16_model = VGG16(
    include_top=False,
    weights='imagenet',
    input_shape=(img_height, img_width, 3)
    )

    model = Sequential()
    model.add(vgg16_model)
    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(train_generator.num_classes, activation='softmax'))

    # Compile the model
    model.compile(
        loss='categorical_crossentropy',
        optimizer='adam',
        metrics=['accuracy']
    )

    # Early stopping callback (configured to monitor the validation loss, wait for 3 epochs with no improvement before stopping training, and restore the best weights seen during training.)
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=10,
        mode='min',
        restore_best_weights=True)
    
    checkpoint = ModelCheckpoint(
        filepath=model_saveas,
        monitor='val_accuracy',
        mode='max',
        save_best_only=True
        )


    history = model.fit(
        train_generator,
        steps_per_epoch=train_generator.samples // batch_size,
        validation_data=val_generator,
        validation_steps=val_generator.samples // batch_size,
        epochs=epochs,
        callbacks=[early_stopping, checkpoint])

    # Print the best results obtained
    best_val_loss = min(history.history['val_loss'])
    best_val_acc = max(history.history['val_accuracy'])
    best_epoch = history.history['val_loss'].index(best_val_loss) + 1
    print(f'Best Validation Loss: {best_val_loss:.4f} - Best Validation Accuracy: {best_val_acc:.4f} - Best Epoch: {best_epoch}')

    model.save(model_saveas)

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


def predict(path):
    try:
        model = load_model('model.h5')
        class_labels = list(train_generator.class_indices.keys())

        img_path = path 
        img = load_img(img_path, target_size=(149, 200))
        img_array = img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array /= 255.

        prediction = model.predict(img_array)

        # class label with the highest confidence
        class_idx = np.argmax(prediction[0])
        class_label = class_labels[class_idx]

        # confidence score for the predicted class
        confidence = prediction[0][class_idx]

        print('This image belongs to:', class_label)
        print('Confidence:', confidence)

    except Exception as e:
        print("Error: {e}")

# train_model()