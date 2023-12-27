import streamlit as st
from PIL import Image
from model.retinaModel import load_retina_model
from utils.image_preprocessing import preprocess_input_image
import numpy as np

model_path = "eye_retina2.h5"
model = load_retina_model(model_path)


title_alignment = """
    <style>
        .e1eexb540 {
            text-align: center;
            color: red;
            font-style: italic;
            margin-left: 40px;
        }
        .stAlert{
            margin-right: 40px;
            margin-left: -15px;
        }
    </style>
"""

def load_and_predict(upload_file):
    img_array = preprocess_input_image(upload_file)
    prediction = model.predict(img_array, batch_size = None)
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
    st.title('Input Retina picture')
    
    col1, col2, col3 = st.columns([1, 1, 1])
    
    with col2:
        st.write('<style>div.row-widget.stRadio > div{flex-direction:row;justify-content: center; margin:1px;}</style>', unsafe_allow_html=True)
        st.write('<style>div.st-bf{flex-direction:column;} div.st-ag{font-weight:bold;padding-left:2px;}</style>', unsafe_allow_html=True)
        
    upload_option = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if upload_option:

        prediction = load_and_predict(upload_option)
        
        col1, col2, col3 = st.columns([1, 1, 1]) 
        with col2:
            st.markdown(title_alignment, unsafe_allow_html=True)
            display_prediction(prediction)

        col1, col2, col3, col4, col5 = st.columns([1.5, 1, 1, 1, 1])
        
        with col2:
            st.image(upload_option, caption='Uploaded Image.', width=300)
    
    with st.sidebar:
        st.markdown("<h1 style='color: black; text-align: center;  margin-top: -30px; margin-bottom:30px; font-weight: bold;'>WWhat do we do here</h1>", unsafe_allow_html=True)
    
if __name__ == "__main__":
    main()