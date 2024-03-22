import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
from tensorflow.keras.preprocessing import image as keras_image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.models import load_model


model = load_model('mobilenetv2_model.h5')


class_labels = ['Aloevera', 'Hibiscus', 'Lemon', 'Mint', 'Kepala']


plant_info_df = pd.read_csv('info_plant.csv')


def preprocess_image(img_path, target_size=(224, 224)):
    img = Image.open(img_path)
    img = img.resize(target_size, Image.BILINEAR)
    img_array = keras_image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    return img_array


def predict_image_class(image):
    test_img = preprocess_image(image)
    predictions = model.predict(test_img)
    predicted_class = np.argmax(predictions)
    predicted_class_label = class_labels[predicted_class]
    return predicted_class_label


st.title("Medicinal Plant Species Detection")


uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image.', use_column_width=True)
    st.write("")

    # Predict button
    if st.button("Predict"):
        # Make prediction
        predicted_plant = predict_image_class(uploaded_file)
        st.success(f"Predicted Leaf: {predicted_plant}")

        # Display plant information from CSV
        plant_info = plant_info_df[plant_info_df['Common Name'] == predicted_plant]
        st.write("Plant Information:")
        for index, row in plant_info.iterrows():
            st.write(f"Common Name: {row['Common Name']}")
            st.write(f"Other Names: {row['Other Names']}")
            st.write(f"Scientific Name: {row['Scientific Name']}")
            st.write(f"About: {row['About']}")
            st.write(f"Medicinal Benefits: {row['Medicinal Benefits']}")
