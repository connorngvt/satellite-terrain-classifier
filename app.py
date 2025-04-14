import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import load_model
import streamlit as st
import numpy as np
from PIL import Image

model = load_model('C:/Users/conno/Development/Individual/python_projects/image_classification/Satellite_Image_Classify.keras')

data_cat = ['cloudy', 'desert', 'green_area', 'water']

img_width = 180
img_height = 180
image = 'Cloudy_Test.jpg'
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    image = image.resize((img_width, img_height))
    img_arr = np.array(image)
    img_bat = tf.expand_dims(img_arr, 0)

    predict = model.predict(img_bat)

    score = tf.nn.softmax(predict)

    st.image(image)
    st.write(f'Terrain in image is {data_cat[np.argmax(score)]}')
    st.write(f'With confidence of {np.max(score) * 100:0.2f}')