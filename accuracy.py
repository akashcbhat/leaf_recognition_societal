import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing import image as keras_image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.models import load_model

# Load your trained model
model = load_model('mobilenetv2_model.h5')

# Define class labels
class_labels = ['Aloevera', 'Hibiscus', 'Lemon', 'Mint', 'Kepala']

# Define path to your testing dataset directory
testing_dir = r'C:\AKASH\mca\Societal Project\testing_dataset\Medicinal plant dataset'

# Initialize dictionaries to store accuracy for each class
class_accuracies = {label: {'total': 0, 'correct': 0} for label in class_labels}

# Initialize lists to store ground truth labels and predicted labels
true_labels = []
predicted_labels = []

# Iterate through each directory in the testing directory
for dir_name in os.listdir(testing_dir):
    # Get the full path to the current directory
    dir_path = os.path.join(testing_dir, dir_name)
    
    # Get the true label for the current directory
    true_label = dir_name
    
    # Initialize variables to count correct predictions for the current class
    total_correct = 0
    total_samples = 0
    
    # Iterate through each image file in the current directory
    for image_file in os.listdir(dir_path):
        # Load and preprocess the image
        img_path = os.path.join(dir_path, image_file)
        img = keras_image.load_img(img_path, target_size=(224, 224))
        img_array = keras_image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)
        
        # Perform inference to get predicted label
        predictions = model.predict(img_array)
        predicted_class = np.argmax(predictions)
        predicted_class_label = class_labels[predicted_class]
        
        # Update counts
        total_samples += 1
        if predicted_class_label == true_label:
            total_correct += 1
            
        # Append true label and predicted label
        true_labels.append(true_label)
        predicted_labels.append(predicted_class_label)
    
    # Store accuracy for the current class
    class_accuracies[true_label]['total'] = total_samples
    class_accuracies[true_label]['correct'] = total_correct

# Calculate accuracy for each class
class_accuracies_percentage = {}
for label in class_labels:
    accuracy = class_accuracies[label]['correct'] / class_accuracies[label]['total'] * 100
    class_accuracies_percentage[label] = accuracy

# Plot accuracy as a bar graph
plt.figure(figsize=(10, 6))
plt.bar(class_accuracies_percentage.keys(), class_accuracies_percentage.values(), color='skyblue')
plt.xlabel('Class Labels')
plt.ylabel('Accuracy (%)')
plt.title('Accuracy for Each Class')
plt.ylim(0, 100)
plt.xticks(rotation=45)
plt.grid(axis='y')
plt.tight_layout()
plt.show()
