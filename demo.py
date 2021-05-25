import cv2
import json
import numpy as np
from PIL import Image, ImageOps
import tensorflow as tf
import tensorflow_hub as hub
from tensorflow.keras.models import Sequential, save_model,load_model
import streamlit as st
import matplotlib.pyplot as plt

st.title("Leaf Disease Prediction")
st.write(" Image classification web app to predict Leaf Disease")
file = st.file_uploader("Please upload an image file", type=["jpg", "png"])

#saved_model_dir = 'F:/my_model.h5'
model = tf.keras.models.load_model(('my_model.h5'),custom_objects={'KerasLayer':hub.KerasLayer})

def load_predict(image_data,model):
    size=(224,224)
    image=ImageOps.fit(image_data,size,Image.ANTIALIAS)
    image=np.asarray(image)
    img=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    img_resize=(cv2.resize(img,dsize=(75,75),interpolation=cv2.INTER_CUBIC))/255.
    probabilities=model.predict(np.asarray([img]))[0]
    class_idx=np.argmax(probabilities)
    return {classes[class_idx]:probabilities[class_idx]}

with open('F:\cat.json', 'r') as f:
    cat_to_name = json.load(f)
    classes = list(cat_to_name.values())

if file is None:
    st.text("Please upload an image file")
else:
    image = Image.open(file)
    st.image(image, use_column_width=True)
    prediction = load_predict(image, model)
    st.write("PREDICTED: class: %s, confidence: %f" % (list(prediction.keys())[0], list(prediction.values())[0]))
    st.write(prediction)
