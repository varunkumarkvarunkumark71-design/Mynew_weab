# Mynew_weab
Mynew_weab
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os

# Define the image dimensions and batch size
IMAGE_WIDTH, IMAGE_HEIGHT = 128, 128
BATCH_SIZE = 32
NUM_CLASSES = 2  # Healthy and Demented

# --- Data Preparation ---
# Create an ImageDataGenerator for data augmentation and normalization
# This helps the model generalize better and prevents overfitting
datagen = ImageDataGenerator(
    rescale=1./255, # Normalize pixel values to [0, 1]
    validation_split=0.2 # Use 20% of the data for validation
)

# Load training data from the 'data' directory
train_generator = datagen.flow_from_directory(
    'data',
    target_size=(IMAGE_WIDTH, IMAGE_HEIGHT),
    batch_size=BATCH_SIZE,
    class_mode='binary',  # 'binary' for two classes (Healthy/Demented)
    subset='training' # Specify training data
)

# Load validation data from the 'data' directory
validation_generator = datagen.flow_from_directory(
    'data',
    target_size=(IMAGE_WIDTH, IMAGE_HEIGHT),
    batch_size=BATCH_SIZE,
    class_mode='binary',
    subset='validation' # Specify validation data
)

# --- Build the CNN Model ---
model = Sequential([
    # First Convolutional Layer
    Conv2D(32, (3, 3), activation='relu', input_shape=(IMAGE_WIDTH, IMAGE_HEIGHT, 3)),
    MaxPooling2D((2, 2)),
    
    # Second Convolutional Layer
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),

    # Third Convolutional Layer
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),

    # Flatten the 3D feature maps to 1D vector
    Flatten(),

    # Fully Connected (Dense) Layers
    Dense(128, activation='relu'),
    Dense(1, activation='sigmoid') # Sigmoid for binary classification
])

# --- Compile the Model ---
model.compile(
    optimizer='adam',
    loss='binary_crossentropy', # Appropriate loss for binary classification
    metrics=['accuracy']
)

# --- Train the Model ---
# You can adjust the number of epochs. 5-10 is a good starting point for a Hackathon.
EPOCHS = 10
history = model.fit(
    train_generator,
    epochs=EPOCHS,
    validation_data=validation_generator
)

# --- Save the Trained Model ---
model.save('dementia_detector_model.h5')
print("Model saved as 'dementia_detector_model.h5'")

# To see the labels the model learned:
print("Class indices:", train_generator.class_indices)


import os
from flask import Flask, render_template, request, jsonify
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image

app = Flask(__name__)

# Load the pre-trained model
MODEL_PATH = 'dementia_detector_model.h5'
model = tf.keras.models.load_model(MODEL_PATH)

# Define the labels and target image size
LABELS = {0: 'Healthy', 1: 'Demented'}
IMAGE_WIDTH, IMAGE_HEIGHT = 128, 128

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})

    if file:
        try:
            # Preprocess the image
            img = image.load_img(file, target_size=(IMAGE_WIDTH, IMAGE_HEIGHT))
            img_array = image.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)
            img_array /= 255.0  # Normalize to [0, 1]

            # Make a prediction
            prediction = model.predict(img_array)
            probability = prediction[0][0]

            # Determine the class and confidence
            if probability > 0.5:
                predicted_class = LABELS[1] # Demented
                confidence = float(probability) * 100
            else:
                predicted_class = LABELS[0] # Healthy
                confidence = float(1 - probability) * 100

            return jsonify({
                'prediction': predicted_class,
                'confidence': f'{confidence:.2f}%'
            })
            
        except Exception as e:
            return jsonify({'error': str(e)})

if __name__ == '__main__':
    # You can set debug=True for development
    app.run(debug=True)
