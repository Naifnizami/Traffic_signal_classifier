import kagglehub
import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator  # Updated import
from tensorflow.keras.utils import load_img, img_to_array
import os

# Download and Set Up Dataset
def setup_dataset():
    print("Downloading dataset from Kaggle...")
    path = kagglehub.dataset_download("ahemateja19bec1025/traffic-sign-dataset-classification")
    print("Path to dataset files:", path)
    
    # The correct directory based on the folder structure you found
    data_dir = os.path.join(path, "traffic_Data", "DATA")  # Adjusted path to "DATA"
    
    # Check if the directory exists
    if not os.path.exists(data_dir):
        print(f"Error: {data_dir} does not exist!")
        exit(1)
    
    return data_dir

# Load Dataset
def load_data(data_dir):
    datagen = ImageDataGenerator(rescale=1.0/255, validation_split=0.2)
    train_data = datagen.flow_from_directory(
        data_dir,
        target_size=(64, 64),
        batch_size=32,
        class_mode='categorical',
        subset='training'
    )
    val_data = datagen.flow_from_directory(
        data_dir,
        target_size=(64, 64),
        batch_size=32,
        class_mode='categorical',
        subset='validation'
    )
    return train_data, val_data

# Build Model
def build_model():
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dense(58, activation='softmax')  # Update to 58 classes
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Train Model
def train_model(model, train_data, val_data, epochs=10):
    model.fit(train_data, validation_data=val_data, epochs=epochs)
    return model

# Preprocess a single image for prediction
def preprocess_image(image_path):
    img = load_img(image_path, target_size=(64, 64))  # Resize to match model input
    img_array = img_to_array(img) / 255.0            # Normalize to [0, 1]
    img_array = np.expand_dims(img_array, axis=0)    # Add batch dimension
    return img_array

# Load labels from CSV
def load_labels(path):
    labels_file = os.path.join(path, "labels.csv")
    if os.path.exists(labels_file):
        labels_df = pd.read_csv(labels_file)
        label_mapping = dict(zip(labels_df["ClassId"], labels_df["SignName"]))
        return label_mapping
    return None

# Evaluate Test Data
def evaluate_test_data(model, test_dir, label_mapping):
    print("Evaluating on test data...")
    test_files = [os.path.join(test_dir, f) for f in os.listdir(test_dir) if f.endswith('.png')]
    results = []

    for test_file in test_files:
        img_array = preprocess_image(test_file)
        prediction = model.predict(img_array, verbose=0)
        predicted_class = np.argmax(prediction, axis=1)[0]  # Get class with highest probability
        readable_label = label_mapping.get(predicted_class, "Unknown") if label_mapping else predicted_class
        results.append((test_file, readable_label))
    
    return results

# Example Usage
if __name__ == "__main__":
    # Setup and load the dataset
    data_dir = setup_dataset()
    train_data, val_data = load_data(data_dir)

    # Build and train the model
    model = build_model()
    model = train_model(model, train_data, val_data, epochs=5)

    # Save the trained model
    model.save("traffic_signal_classifier.keras")
    print("Model trained and saved successfully!")

    # Evaluate on test data
    test_dir = os.path.join(data_dir, "..", "TEST")  # Adjust path to locate TEST folder
    label_mapping = load_labels(data_dir)  # Load label mapping from labels.csv
    test_results = evaluate_test_data(model, test_dir, label_mapping)

    # Display a few test results
    print("Test Results (Sample):")
    # Example of using label_mapping to convert numeric predictions to readable labels
    for file, pred_class in test_results[:10]:  # Show first 10 predictions
        readable_label = label_mapping.get(pred_class, "Unknown") if label_mapping else pred_class
        print(f"File: {file}, Predicted Class: {readable_label}")




