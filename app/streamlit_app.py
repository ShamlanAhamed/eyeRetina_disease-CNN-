import streamlit as st
from PIL import Image
from model.retinaModel import load_retina_model
from utils.image_preprocessing import preprocess_input_image
import numpy as np

model_path = "model/eye_retina.h5"
model = load_retina_model(model_path)

def load_and_predict(upload_file):
    img_array = preprocess_input_image(upload_file)
    prediction = model.predict(img_array, batch_size = None, step = 1)
    return prediction


def display_prediction(prediction):
    # Interpret the prediction
    if prediction[0][0] > 0.5:
        
        st.header('Healthy')
        st.warning('Good job')
        st.balloons()
    else:
        print("Predicted: Diseased")
        st.header('Diseased')
        st.success('please consult doctor')

def main():        
    st.title('Input | Capture faces to detect mask')    
    
if __name__ == "__main__":
    main()