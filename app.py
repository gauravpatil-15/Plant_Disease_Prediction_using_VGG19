import streamlit as st
import cv2
import numpy as np
import pickle
import tensorflow as tf
import mysql.connector

model = tf.keras.models.load_model('best_model28.h5')
classes = pickle.load(open('classes.pkl','rb'))


def disease_details(result):
    cnx = mysql.connector.connect(host="localhost",
                                user='root',
                                password="Gaurav@12345",
                                database='Disease_Treatments')
    
    cursor = cnx.cursor()
    query = f'SELECT * FROM treatments WHERE ID = "{np.argmax(result)}" '

    cursor.execute(query)
    details = cursor.fetchall() 

    return details


st.markdown("<h1 style='text-align: center; color: green;'>Plant Disease Prediction System</h1>", unsafe_allow_html=True)
st.markdown("<h4 style='text-align: center;'>Welcome to Plant Disease Prediction System...😊🪴</h4>", unsafe_allow_html=True)
st.divider()
st.text("")
st.subheader("This is a two step process :")
st.text("1.Upload the image of the plant..")
st.text("2.Then hit the Pridict button")

st.divider()


option = st.selectbox("Choose one of the Options :", ("Select", "File Uploader", "Camera"))
st.divider()

if option == "File Uploader":
    

    uploaded_files = st.file_uploader("Upload Plant Images : ", accept_multiple_files=True, type=['png', 'jpg', 'jpeg'])
    
    col1, col2, col3  = st.columns(3)

    with col2 :
        submit_img = st.button("Predict")

    st.divider()


    if submit_img and uploaded_files is not None :

        for uploaded_file in uploaded_files:

            bytes_data = uploaded_file.read()
            st.write("Filename:", uploaded_file.name)

            img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
            img2 = cv2.resize(img,(256,256))

            gray = cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)  # Converting color image to gray image..
            ret, thresh = cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
            
            col1, col2, col3 = st.columns(3)
            with col1 :
                st.write("Original Image : ")
                st.image(img2)

            with col2 :
                st.write("Grayscale Image : ")
                st.image(gray)

            with col3 :
                st.write("After OTSU Binarization : ")
                st.image(thresh)


            img3 = np.reshape(img2,[1,256,256,3])
            result = model.predict(img3)
            key = np.argmax(result)
            st.divider()
            # st.write(f"The image belongs to {classes[np.argmax(result)]} class")

            details = disease_details(result)

            if(key == 2 or key == 4 or key == 9 or key == 12):
                st.markdown(f"This {details[0][1]} plant is **healthy**..👍🪴")
            else:
                st.markdown(f"<h4 style = 'text: bold;'> This {details[0][1]} plant has {details[0][2]} disease..🙁</h4", unsafe_allow_html=True)
                st.markdown("<h3 style = 'color: green;'>Disease Discription :</h3>", unsafe_allow_html=True)
                st.write(details[0][3])
                st.markdown("<h3 style = 'color: green;'>Treatment :</h3>", unsafe_allow_html=True)
                st.write(details[0][4])
            st.divider()
            st.divider()


elif option == "Camera":
    
    camera_pic = st.camera_input("Take a picture")

    col1, col2, col3  = st.columns(3)

    with col2 :
        submit_img = st.button("Predict")

    st.divider()

    if submit_img and camera_pic is not None:
        bytes_data = camera_pic.read()

        img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
        img2 = cv2.resize(img,(256,256))

        gray = cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)  # Converting color image to gray image..
        ret, thresh = cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
        
        col1, col2, col3 = st.columns(3)
        with col1 :
            st.write("Original Image : ")
            st.image(img2)

        with col2 :
            st.write("Grayscale Image : ")
            st.image(gray)

        with col3 :
            st.write("After OTSU Binarization : ")
            st.image(thresh)


        img3 = np.reshape(img2,[1,256,256,3])
        result = model.predict(img3)
        key = np.argmax(result)
        st.divider()
        # st.write(f"The image belongs to {classes[np.argmax(result)]} class")

        details = disease_details(result)

        if(key == 2 or key == 4 or key == 9 or key == 12):
            st.markdown(f"This {details[0][1]} plant is **healthy**..👍🪴")
        else:
            st.markdown(f"<h4 style = 'text: bold;'> This {details[0][1]} plant has {details[0][2]} disease..🙁</h4", unsafe_allow_html=True)
            st.markdown("<h3 style = 'color: green;'>Diasese Discription :</h3>", unsafe_allow_html=True)
            st.write(details[0][3])
            st.markdown("<h3 style = 'color: green;'>Treatment :</h3>", unsafe_allow_html=True)
            st.write(details[0][4])

elif option == "Select":
    pass



    
    
    


# {
#   "0": "Apple___Apple_scab",
#   "1": "Apple___Cedar_apple_rust",
#   "2": "Apple___healthy",
#   "3": "Corn_(maize)__Common_rust",
#   "4": "Corn_(maize)___healthy",
#   "5": "Grape___Black_rot",
#   "6": "Grape___healthy",
#   "7": "Potato___Early_blight",
#   "8": "Potato___Late_blight",
#   "9": "Potato___healthy",
#   "10": "Tomato___Bacterial_spot",
#   "11": "Tomato___Leaf_Mold",
#   "12": "Tomato___healthy"
# }