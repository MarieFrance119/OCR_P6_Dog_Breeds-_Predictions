# -*- coding: utf-8 -*-
"""
To run the app, go to the terminal and write
streamlit run app.py

Then the app is available at :
http://localhost:8501
"""

# Data manipulation
import numpy as np 

# Image manipulation
from PIL import Image

# Keras tools
from keras.preprocessing import image
from keras.models import load_model
from keras.applications.mobilenet_v2 import preprocess_input

# data persistence
import joblib

# Streamlit
import streamlit as st

# Load model
model = load_model("./models/opt_model_MobileNetV2.h5")

# Load List of breeds
breeds_list = joblib.load("./models/list_breeds.joblib")

# String with the 12 dog breeds
breeds_str = (
    "American_Staffordshire_terrier, Australian_terrier, beagle, Brittany_spaniel, \n"
    "cocker_spaniel, English_setter, French_bulldog, German_shepherd, \n"
    "golden_retriever, Labrador_retriever, malinois, Staffordshire_bullterrier\n"
    )


def breed_prediction(input_img, list_breeds) : 
    """
        Function to get breed prediction and display resized original image

        Arguments :
            - input_img : image to load
            - list_breeds : list with the breeds (= class names)

        Return : 
            - predicted breed and the percentage of this prediction
    """
  
    # setting dimension of the resize
    SIZE = 224

    # resize image
    resized_img = input_img.resize((SIZE, SIZE))
    
    # Array
    img_array = image.img_to_array(resized_img)

    # Tensor
    img_tensor = img_array.reshape((1,)+img_array.shape)
    img_tensor_preproc = preprocess_input(img_tensor)

    # Prediction
    pred = model.predict(img_tensor_preproc)

    # Predicted breed
    pred_label = list_breeds[np.argmax(pred)]

    # Prediction percentage
    pred_perc = round(np.max(pred)*100,1)

    return pred_label, pred_perc


def main():
    
    # Background
    media = Image.open('./media/dogs.jpeg' )
    st.image(media, caption = "image from https://www.istockphoto.com")

    #Title
    st.title("Dog Breed Prediction from a Picture with MobileNetV2 Model")

    # Text
    st.text("Limited to 12 breeds among : \n" + breeds_str)

    # File uploader
    file = st.file_uploader("Load a picture of your dog")

    # Initialization of placeholders
    img_placeholder = st.empty()
    success = st.empty()
    submit_placeholder = st.empty()
    submit=False

    # When uploading image
    if file is not None :
        with st.spinner("Picture loading..."):  
            img = Image.open(file)
            img_placeholder.image(img)

        # Submit button creation
        submit = submit_placeholder.button("Launch prediction")

    # When clicking on submit button
    if submit :
        with st.spinner('Searching...'):    
            submit_placeholder.empty()

            # Get prediction and its percentage
            pred, perc = breed_prediction(img, breeds_list)

            # Display results
            success.success(
                "Predicted dog breed : {} with {}% of chances".format(
                    pred, perc))

if __name__ == "__main__":
    main()
