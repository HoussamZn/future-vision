import streamlit as st

# Set a custom title and description
def home():
    st.title("Welcome to Our Platform!")
    st.write("Get started with our platform by watching these quick tutorials.")
    st.write("---")

    st.write("## Data manipulation")
    st.write("This platform is designed to help you with your data manipulation tasks.")
    st.write("You can perform the following operations:")
    st.write("* Import or create your data")
    st.write("* Take a clear look at your data using the data explorer")
    st.write("* modify your data (drop columns, add rows, etc.)")
    st.write("* filter your data")
    st.write("* Export your data")

    video_file = open("videos/1.mp4", "rb")
    video_bytes = video_file.read()
    st.video(video_bytes)

    st.write("---")
    st.write("## Data Pre-Proccessing")
    st.write("This platform is designed to help you with your data Pre-Proccessing Task.")
    st.write("You can perform the following operations to process your Data :")
    st.write("* Handle missing values")
    st.write("* Handle categorical data")
    st.write("* Balancing your data")
    st.write("* visualize your data")


    video_file = open("videos/3.mp4", "rb")
    video_bytes = video_file.read()
    st.video(video_bytes)

    st.write("---")
    st.write("## Training Models & Predictions")
    st.write("This platform is designed to help you with your training models and make your prediction.")
    st.write("You can perform the following operations to train your model and make predictions :")
    st.write("* Import a trained model or train a new one")
    st.write("* adjust the model parameters")
    st.write("* make predictions on new data")
    st.write("* export your model to use it in your applications later on")

    video_file = open("videos/4.mp4", "rb")
    video_bytes = video_file.read()
    st.video(video_bytes)