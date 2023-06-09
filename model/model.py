import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator,load_img, img_to_array
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# parameters for data loading and image preprocessing
batch_size = 64
img_height = 149
img_width = 200
epochs = 100
train_dir = 'path/to/dataset'
model_saveas = "model.h5"

train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    validation_split=0.2)  # split 20% of data for validation

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
    model = keras.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu',
                    input_shape=(img_height, img_width, 3)),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Flatten(),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(14, activation='softmax')
    ])

    model.compile(optimizer='adam',
                loss='categorical_crossentropy',
                metrics=['accuracy'])
    
    # Early stopping callback (configured to monitor the validation loss, wait for 10 epochs with no improvement before stopping training, and restore the best weights seen during training.)
    early_stopping = EarlyStopping(
        monitor='val_accuracy',
        patience=10,
        mode='max',
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

    # Plotting the training and validation loss and accuracy
    sns.set_style("darkgrid")
    plt.figure(figsize=(8, 6))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Training and Validation Metrics')
    plt.xlabel('Epoch')
    plt.ylabel('Metric Value')
    plt.legend()
    plt.savefig('metric.png')

    model.save(model_saveas)

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
# predict(path)