import streamlit as st
from keras.models import load_model
from keras.preprocessing.image import load_img, img_to_array
import numpy as np
import os
import requests
from PIL import Image
import io
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

import os

email_user = os.environ.get('GMAIL_USER')
email_password = os.environ.get('GMAIL_PASSWORD')

if email_user is None or email_password is None:
    print("Environment variables not set")
else:
    print("Environment variables set")


def send_email(email, subject, content):
    msg = MIMEMultipart()
    msg['From'] = os.getenv('GMAIL_USER')
    msg['To'] = email
    msg['Subject'] = subject
    msg.attach(MIMEText(content, 'plain'))

    try:
        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.starttls()
        server.login(os.getenv('GMAIL_USER'), os.getenv('GMAIL_PASSWORD'))
        text = msg.as_string()
        server.sendmail(os.getenv('GMAIL_USER'), email, text)
        server.quit()
        print("Email sent successfully.")
    except Exception as e:
        print(e)

model = load_model('final_model.h5')  # load model at the beginning
UPLOAD_FOLDER = 'uploaded_images'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Dictionary mapping numerical indices to corresponding lesion types
lesion_type_dict_idx = {
    0: 'Actinic keratoses',
    1: 'Basal Cell Casrsinoma',
    2: 'Benign keratosis-like lesions ',
    3: 'Dermatofibroma',
    4: 'Melanocytic nevi',
    5: 'Melanoma',
    6: 'Vascular leisons'

    # Index: 2, Cell Type: Benign keratosis-like lesions 
    # Index: 4, Cell Type: Melanocytic nevi
    # Index: 3, Cell Type: Dermatofibroma
    # Index: 5, Cell Type: Melanoma
    # Index: 6, Cell Type: Vascular lesions
    # Index: 1, Cell Type: Basal cell carcinoma
    # Index: 0, Cell Type: Actinic keratoses
    
}

# List to store file paths of images uploaded by the user
recent_images = []

def save_uploadedfile(uploadedfile):
    with open(os.path.join(UPLOAD_FOLDER,uploadedfile.name),"wb") as f:
        f.write(uploadedfile.getbuffer())
    return st.success("Saved File:{} to {}".format(uploadedfile.name,UPLOAD_FOLDER))

st.title("SpotDetection: Skin Cancer Detection App")
st.write("This app predicts the type of skin lesion in an uploaded image")

st.header("Upload a file or provide a URL")
method = st.radio("Upload method:", ("Upload a file", "Provide a URL", "Use a recent image"))

file = None  # Declare 'file' before the if-else block

if method == "Upload a file":
    file = st.file_uploader("Please upload an image", type=["jpg", "png"])
    if file is None:
        st.text("Please upload an image file")
    else:
        save_uploadedfile(file)
        image_location = os.path.join(UPLOAD_FOLDER, file.name)
        recent_images.append(image_location)  # Add the file path to the recent images list
elif method == "Provide a URL":
    url = st.text_input("Please input image url", "")
    if url == "":
        st.text("Please input url")
    else:
        response = requests.get(url)
        file = io.BytesIO(response.content)
        image_location = os.path.join(UPLOAD_FOLDER, "temp.jpg")
        with open(image_location, "wb") as f:
            f.write(file.read())
        recent_images.append(image_location)  # Add the file path to the recent images list
elif method == "Use a recent image":
    if len(recent_images) > 0:
        image_location = st.selectbox("Please select an image", options=recent_images)
        file = open(image_location, "rb")
    else:
        st.text("No recent images available")

# Keep only the last 5 images
recent_images = recent_images[-5:]

if file is not None:
    img = load_img(image_location, target_size=(75, 100))
    img = img_to_array(img) / 255.0  # Normalize to [0,1]
    img = np.expand_dims(img, axis=0)
    pred = model.predict(img)
    pred_class = np.argmax(pred, axis=1)
    
    st.image(image_location, use_column_width=True)
    st.write(f"Prediction: {lesion_type_dict_idx[pred_class[0]]}")
    st.write(f"Probability: {np.max(pred[0])}")

    # Add a box for the user to input their email
    email = st.text_input("Please enter your email", "")

    # Check if the user has provided an email before attempting to send
    if email != "":
        if st.button("Send email"):
            send_email(email, 'Skin Lesion Analysis Results', f'Your skin lesion has been classified as {lesion_type_dict_idx[pred_class[0]]}')

# Display the last 5 images uploaded by the user
st.header("Recent Images")
for image_path in recent_images:
    st.image(image_path, use_column_width=True)
