import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import random
import tensorflow as tf
import cv2
import matplotlib.pyplot as plt

st.set_page_config(page_title="🎀 Image App", layout="wide")

# Session state navigation
if "page" not in st.session_state:
    st.session_state.page = "welcome"

# ========== CSS PINK + RIBBON ANIMATION 🎀 ==========
st.markdown("""
<style>
.stApp {
    background-color: #ffe6f2 !important;
}
section[data-testid="stSidebar"] {
    background-color: #ffe6f2 !important;
}
h1, label, p, .stRadio, .stMarkdown {
    color: #b30059 !important;
}
.main-title, .welcome-title {
    font-size: 38px;
    font-weight: bold;
    text-align: center;
    color: #b30059;
    z-index: 10;
}
div[data-testid="stFileUploader"] {
    background-color: #ffd8e9 !important;
    border: 3px dashed #b30059 !important;
    border-radius: 12px;
    padding: 20px;
}
.stButton>button {
    background-color: #ff99c8 !important;
    color: white !important;
    border-radius: 10px;
    border: 2px solid #b30059;
}
.ribbon {
    position: fixed;
    top: -10vh;
    opacity: 0.30;
    z-index: 1 !important;
    animation: fall 7s linear infinite;
}
@keyframes fall {
    0% { transform: translateY(-10vh) rotate(0deg); }
    100% { transform: translateY(110vh) rotate(360deg); }
}
</style>
""", unsafe_allow_html=True)

# ====== Falling Pita 🎀 ======
def falling_ribbons():
    ribbons = ""
    for _ in range(70):
        x = random.randint(0, 100)
        size = random.randint(20, 45)
        speed = random.uniform(3, 7)
        ribbons += f"""
        <span class='ribbon'
              style='left:{x}vw;
              font-size:{size}px;
              animation-duration:{speed}s;'>🎀</span>"""
    st.markdown(ribbons, unsafe_allow_html=True)

falling_ribbons()

# ========== Welcome Page ==========
if st.session_state.page == "welcome":
    st.markdown("<h1 class='welcome-title'>🎀 WELCOME TO NAI'S APPLICATION 🎀</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align:center;'>Image Detection & Classification 🎀</p>", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    if st.button("START THE APPLICATION 🎀", use_container_width=True):
        st.session_state.page = "main"
        st.rerun()

# ========== MAIN PAGE ==========
elif st.session_state.page == "main":

    st.markdown("<h1 class='main-title'>🎀 IMAGE APP - CLASSIFICATION & DETECTION 🎀</h1>", unsafe_allow_html=True)

    uploaded_file = st.file_uploader("🎀 Drag & Drop Image Here 🎀", type=["jpg", "jpeg", "png"])

    # ====== Load Model YOLO + H5 ======
    @st.cache_resource
    def load_model():
        yolo = YOLO("C:/Users/Lenovo/Downloads/data big lap 5 uts/model/best.pt")
        classifier = tf.keras.models.load_model("C:/Users/Lenovo/Downloads/data big lap 5 uts/model/best.h5")
        return yolo, classifier

    yolo_model, classifier_model = load_model()

    col1, col2 = st.columns([1, 2])

    with col1:
        mode = st.radio("🎀 Pilih Mode:",
                        ["Object Detection (YOLO)", "Image Classification (.h5 Model)"])

    # ====== Jika image telah upload ======
    if uploaded_file:
        img = Image.open(uploaded_file).convert("RGB")

        with col2:
            st.image(img, caption="🎀 Image Uploaded", use_column_width=True)

            if st.button("🎀 Process 🎀"):

                # ====== DETEKSI YOLO ======
                if mode == "Object Detection (YOLO)":
                    results = yolo_model.predict(np.array(img))
                    result_img = results[0].plot()
                    st.image(result_img, caption="🎀 YOLO Detection Result ✅", use_column_width=True)

                # ====== KLASIFIKASI ======
                else:
                    st.subheader("🎀 Hasil Klasifikasi Gambar")

                    input_shape = classifier_model.input_shape[1:3]
                    img_resized = img.resize(input_shape)
                    img_array = np.expand_dims(np.array(img_resized) / 255.0, axis=0)
                    prediction = classifier_model.predict(img_array)

                    # Multi-class ✅
                    if prediction.shape[1] > 1:
                        class_index = int(np.argmax(prediction))
                        prob = float(np.max(prediction))

                        st.metric("🎀 Kelas Prediksi", f"Kelas {class_index}")
                        st.metric("🎀 Probabilitas", f"{prob:.2%}")

                        fig, ax = plt.subplots()
                        ax.bar(range(len(prediction[0])), prediction[0])
                        ax.set_xticks(range(len(prediction[0])))
                        ax.set_xticklabels([f"Kelas {i}" for i in range(len(prediction[0]))])
                        ax.set_ylabel("Probabilitas")
                        ax.set_title("🎀 Distribusi Probabilitas Prediksi")
                        st.pyplot(fig)

                    # Binary classification ✅
                    else:
                        prob = float(prediction[0][0])
                        label = "Stopwatch 🎀" if prob > 0.5 else "Jam 🎀"

                        st.metric("🎀 Prediksi Kelas", label)
                        st.metric("🎀 Probabilitas", f"{prob:.2%}")
