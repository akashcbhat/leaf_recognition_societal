import pandas as pd
import numpy as np
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
from sklearn.model_selection import train_test_split
from google.colab import drive


drive.mount('/content/drive')

data_dir = '/content/drive/My Drive/dataset_reduced'

image_size = (224, 224)
batch_size = 32

file_paths = []
labels = []

for class_label in os.listdir(data_dir):
    class_dir = os.path.join(data_dir, class_label)
    for img_name in os.listdir(class_dir):
        file_paths.append(os.path.join(class_dir, img_name))
        labels.append(class_label)

df = pd.DataFrame({'file_path': file_paths, 'label': labels})

# Convert label column to strings
df['label'] = df['label'].astype(str)

# Split data into training and validation sets
train_df, valid_df = train_test_split(df, test_size=0.2, random_state=42)

# Data preprocessing and augmentation using ImageDataGenerator
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

valid_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_dataframe(
    train_df,
    x_col='file_path',
    y_col='label',
    target_size=image_size,
    batch_size=batch_size,
    class_mode='categorical'
)

valid_generator = valid_datagen.flow_from_dataframe(
    valid_df,
    x_col='file_path',
    y_col='label',
    target_size=image_size,
    batch_size=batch_size,
    class_mode='categorical'
)

# Define the number of classes
num_classes = len(df['label'].unique())

# Load MobileNetV2 pre-trained model
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))


x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
predictions = Dense(num_classes, activation='softmax')(x)


model = Model(inputs=base_model.input, outputs=predictions)


for layer in base_model.layers:
    layer.trainable = False

# Compile the model
model.compile(optimizer=Adam(),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
history = model.fit(train_generator,
                    epochs=50,
                    validation_data=valid_generator)

# Save the model
model.save('mobilenetv2_model.h5')
