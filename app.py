import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# 1. Setup the Web Page Profile
st.set_page_config(page_title="Plant Disease Diagnosis via CNN", page_icon="🌿")
st.title("🌿 Plant Disease Diagnosis via CNN")
st.write("This Web App for the ML-AI Major Project titled \"Automated Plant Disease Diagnosis Using Convolutional Neural Networks on Leaf Images\" is a record of an original piece of work done by Nilanjan Bhattacharyya (Enrolment No: A992972400099(el)) for the partial fulfillment of the MCA program at Amity University Online.")

# 2. Load the trained model (Cached so it doesn't reload every time you click a button)
@st.cache_resource
def load_model():
    return tf.keras.models.load_model('plant_disease_model.keras')

model = load_model()

# 3. Define the plant classes (REPLACE THIS LIST WITH YOUR PRINTED OUTPUT FROM STEP 3)
CLASS_NAMES = ['Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy', 
               'Blueberry___healthy', 
               'Cherry_(including_sour)___Powdery_mildew', 'Cherry_(including_sour)___healthy', 
               'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 'Corn_(maize)___Common_rust_', 'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy', 
               'Grape___Black_rot', 'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Grape___healthy', 
               'Orange___Haunglongbing_(Citrus_greening)', 
               'Peach___Bacterial_spot', 'Peach___healthy', 
               'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 
               'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy', 
               'Raspberry___healthy', 
               'Soybean___healthy', 
               'Squash___Powdery_mildew', 
               'Strawberry___Leaf_scorch', 'Strawberry___healthy', 
               'Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold', 'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite', 'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus', 'Tomato___healthy'
               ]

# 4. Create the File Uploader
file = st.file_uploader("Upload a leaf image (JPG/PNG) and click 'Diagnose Leaf'", type=["jpg", "jpeg", "png"])

if file is not None:
    # Display the uploaded image
    image = Image.open(file)
    st.image(image, caption="Uploaded Botanical Specimen", use_column_width=True)
    
    # 5. Prediction Logic
    if st.button("Diagnose Leaf"):
        with st.spinner("Analyzing visual textures and patterns..."):
            # Preprocess the image to match the CNN's expected input (256x256 pixels)
            img = image.resize((256, 256))
            img_array = tf.keras.utils.img_to_array(img)
            img_array = tf.expand_dims(img_array, 0) # Create a batch of 1

            # Make the prediction
            predictions = model.predict(img_array)
            
            # Extract the highest probability
            predicted_class = CLASS_NAMES[np.argmax(predictions[0])]
            confidence = round(100 * (np.max(predictions[0])), 2)

            # 6. Output the Results
            st.success(f"**Diagnosis:** {predicted_class}")
            st.info(f"**Confidence Level:** {confidence}%")