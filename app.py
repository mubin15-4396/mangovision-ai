
import streamlit as st
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

from src.prediction import load_model, preprocess_image, predict_image
from src.gradcam import generate_gradcam

st.set_page_config(
    page_title="MangoVision AI",
    page_icon="🥭",
    layout="wide"
)

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "MangoLite_CNN_v2_100K_Best.keras"

st.markdown("""
<style>
.main {
    background-color: #f7faf5;
}
.block-container {
    padding-top: 2rem;
}
.hero {
    padding: 25px;
    border-radius: 15px;
    background: linear-gradient(120deg, #e4f4d7, #f5fbe9);
    text-align: center;
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="hero">
    <h1>🥭 MangoVision AI</h1>
    <h3>Intelligent Mango Leaf Disease Classification</h3>
    <p>See the Leaf. Understand the Disease.</p>
</div>
""", unsafe_allow_html=True)

@st.cache_resource
def get_model():
    return load_model(str(MODEL_PATH))

st.sidebar.title("🥭 MangoVision AI")
st.sidebar.write("Powered by MangoLite-CNN v2")
st.sidebar.info(
    "An explainable AI research application "
    "for mango leaf disease classification."
)

st.subheader("Upload a Mango Leaf Image")

uploaded_file = st.file_uploader(
    "Choose a leaf image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    st.image(image, caption="Uploaded Mango Leaf", width=350)

    if st.button("Analyze Leaf", type="primary"):

        with st.spinner("Analyzing leaf image..."):

            model = get_model()
            processed_image = preprocess_image(image)
            result = predict_image(model, image)

            predicted_class = result["predicted_class"]
            confidence = result["confidence"]

            st.success("Analysis completed!")

            st.subheader("Prediction Result")

            col1, col2 = st.columns(2)

            with col1:
                st.metric("Predicted Class", predicted_class)

            with col2:
                st.metric(
                    "Prediction Confidence",
                    f"{confidence * 100:.2f}%"
                )

            st.subheader("Class Probability Distribution")

            st.bar_chart(result["probabilities"])

            st.subheader("Explainable AI — Grad-CAM")

            class_index = int(np.argmax(
                list(result["probabilities"].values())
            ))

            heatmap = generate_gradcam(
                model,
                processed_image,
                class_index=class_index
            )

            fig, axes = plt.subplots(1, 3, figsize=(15, 5))

            axes[0].imshow(image)
            axes[0].set_title("Original Leaf")

            axes[1].imshow(heatmap, cmap="jet")
            axes[1].set_title("Grad-CAM Heatmap")

            axes[2].imshow(image.resize((224, 224)))
            axes[2].imshow(heatmap, cmap="jet", alpha=0.4)
            axes[2].set_title("Grad-CAM Overlay")

            for ax in axes:
                ax.axis("off")

            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

st.divider()

st.caption(
    "MangoVision AI | Research Prototype | MangoLite-CNN v2"
)
