import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import model_from_json
from PIL import Image
import numpy as np

def load_model(json_file_path, weights_file_path):
    with open(json_file_path, 'r') as json_file:
        model_json = json_file.read()
    model = model_from_json(model_json)
    model.load_weights(weights_file_path)
    return model

def main():
    st.title("Human Emotion Classification with CNN [Angry, Happy, Sad]")
    st.subheader("Trained on gray scale images dataset from: https://www.kaggle.com/datasets/muhammadhananasghar/human-emotions-datasethes/data")

    st.write("Upload an image")

    # Upload an image file
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)

        st.image(image, caption='Uploaded Image', use_column_width=True)

        image = image.convert('RGB') 

        im = tf.constant(image, dtype="float32")

        im=tf.expand_dims(im, axis=0)

        class_names =["angry", "happy", "sad"]

        # Load the model
        model = load_model('model.json', 'model.weights.h5')

        prediction = tf.argmax(model.predict(im), axis=-1).numpy()

        result = class_names[prediction[0]]

        # Display the prediction
        st.write(f"Predicted Emotion: {result}")

if __name__ == "__main__":
    main()





