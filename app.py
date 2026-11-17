
import streamlit as st
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

from src.prediction import load_model, preprocess_image, predict_image
from src.gradcam import generate_gradcam


# =========================================================
# 1. PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="MangoVision AI",
    page_icon="🥭",
    layout="wide",
    initial_sidebar_state="expanded"
)

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "MangoLite_CNN_v2_100K_Best.keras"


# =========================================================
# 2. CUSTOM GREEN THEME
# =========================================================

st.markdown("""
<style>

@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

.stApp {
    background-color: #F7F6EF;
}

/* Main layout */
.block-container {
    padding-top: 2rem;
    padding-bottom: 3rem;
    max-width: 1450px;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background-color: #123D2A;
}

[data-testid="stSidebar"] * {
    color: #F7F6EF;
}

[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {
    color: #D5E5D7;
}

/* Hero section */
.hero {
    background: linear-gradient(120deg, #123D2A, #286344);
    padding: 38px 35px;
    border-radius: 20px;
    margin-bottom: 28px;
    color: white;
    box-shadow: 0 8px 24px rgba(18, 61, 42, 0.13);
}

.hero h1 {
    color: white;
    font-size: 42px;
    font-weight: 800;
    margin-bottom: 5px;
}

.hero p {
    color: #D5E8D8;
    font-size: 16px;
    margin-bottom: 0;
}

/* Section headings */
.section-title {
    color: #174D35;
    font-size: 23px;
    font-weight: 800;
    margin-top: 15px;
    margin-bottom: 15px;
}

/* Cards */
.info-card {
    background: #FFFFFF;
    border: 1px solid #E2E9DF;
    border-radius: 16px;
    padding: 22px;
    min-height: 120px;
    box-shadow: 0 4px 14px rgba(18, 61, 42, 0.045);
}

.info-label {
    color: #6D8271;
    font-size: 12px;
    font-weight: 700;
    letter-spacing: 1px;
    text-transform: uppercase;
}

.info-value {
    color: #174D35;
    font-size: 27px;
    font-weight: 800;
    margin-top: 9px;
    overflow-wrap: anywhere;
}

/* Upload section */
[data-testid="stFileUploader"] {
    background: #EAF2E8;
    border: 2px dashed #83A98A;
    border-radius: 16px;
    padding: 15px;
}

/* Buttons */
.stButton > button {
    background: #174D35;
    color: white;
    border: none;
    border-radius: 10px;
    padding: 12px 24px;
    font-weight: 700;
    min-height: 48px;
    transition: 0.2s ease;
}

.stButton > button:hover {
    background: #286344;
    color: white;
    border: none;
}

/* Tabs */
.stTabs [data-baseweb="tab"] {
    color: #174D35;
    font-weight: 700;
}

.stTabs [aria-selected="true"] {
    color: #286344;
}

/* Alerts */
[data-testid="stAlert"] {
    border-radius: 12px;
}

/* Footer */
.footer {
    background: #123D2A;
    color: #D5E5D7;
    padding: 20px;
    border-radius: 14px;
    text-align: center;
    margin-top: 35px;
}

.footer strong {
    color: white;
}

</style>
""", unsafe_allow_html=True)


# =========================================================
# 3. LOAD MODEL
# =========================================================

@st.cache_resource
def get_model():
    return load_model(str(MODEL_PATH))


# =========================================================
# 4. SIDEBAR
# =========================================================

with st.sidebar:

    st.markdown("# 🥭 MangoVision AI")

    st.markdown("---")

    st.markdown("### 🌿 About the Platform")

    st.write(
        "An AI-powered research application for "
        "mango leaf disease classification."
    )

    st.markdown("---")

    st.markdown("### 🧠 Deep Learning Model")

    st.success("MangoLite-CNN v2")

    st.write("Input Size: 224 × 224")

    st.write("Number of Classes: 8")

    st.write("Explainability: Grad-CAM")

    st.markdown("---")

    st.markdown("### 🔬 Research Focus")

    st.write("• Lightweight deep learning")

    st.write("• Disease classification")

    st.write("• Explainable AI")

    st.write("• Efficient inference")

    st.markdown("---")

    st.caption("Research Prototype | 2026")


# =========================================================
# 5. HERO HEADER
# =========================================================

st.markdown("""
<div class="hero">
    <h1>🥭 MangoVision AI</h1>
    <p style="font-size:19px; font-weight:600;">
        Intelligent Mango Leaf Disease Classification
    </p>
    <p>See the Leaf. Understand the Disease.</p>
    <br>
    <p>
        <b>POWERED BY MangoLite-CNN v2</b>
        &nbsp; | &nbsp; Explainable Artificial Intelligence
    </p>
</div>
""", unsafe_allow_html=True)


# =========================================================
# 6. IMAGE UPLOAD
# =========================================================

st.markdown(
    '<div class="section-title">🌱 Leaf Analysis Studio</div>',
    unsafe_allow_html=True
)

st.write(
    "Upload a mango leaf image to identify its condition "
    "and explore the model's visual explanation."
)

uploaded_file = st.file_uploader(
    "Choose a mango leaf image",
    type=["jpg", "jpeg", "png"],
    help="Supported formats: JPG, JPEG and PNG"
)


# Reset previous results if a different image is uploaded
if uploaded_file is not None:

    image_bytes = uploaded_file.getvalue()

    if st.session_state.get("image_bytes") != image_bytes:
        st.session_state.pop("analysis_result", None)
        st.session_state["image_bytes"] = image_bytes


if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    preview_col, upload_col = st.columns([1, 1])

    with preview_col:
        st.markdown("#### Uploaded Image")
        st.image(
            image,
            use_container_width=True
        )

    with upload_col:

        st.markdown("#### Ready for Analysis?")

        st.markdown("""
        <div class="info-card">
            <div class="info-label">IMAGE STATUS</div>
            <div class="info-value">Image Loaded ✓</div>
            <p style="color:#6D8271;">
                Your image is ready for AI-powered analysis.
            </p>
        </div>
        """, unsafe_allow_html=True)

        st.write("")

        analyze_button = st.button(
            "🌿 Analyze Mango Leaf",
            type="primary",
            use_container_width=True
        )

        if analyze_button:

            try:
                with st.spinner(
                    "MangoVision AI is analyzing your image..."
                ):

                    model = get_model()

                    processed_image = preprocess_image(image)

                    result = predict_image(model, image)

                    predicted_index = int(
                        np.argmax(
                            list(result["probabilities"].values())
                        )
                    )

                    heatmap = generate_gradcam(
                        model,
                        processed_image,
                        class_index=predicted_index
                    )

                    st.session_state["analysis_result"] = {
                        "result": result,
                        "heatmap": heatmap,
                        "image": image.copy()
                    }

                st.success("Leaf analysis completed successfully!")

            except Exception as e:
                st.error("Analysis could not be completed.")
                st.exception(e)


# =========================================================
# 7. DISPLAY PREDICTION RESULTS
# =========================================================

if "analysis_result" in st.session_state:

    analysis = st.session_state["analysis_result"]

    result = analysis["result"]
    heatmap = analysis["heatmap"]
    original_image = analysis["image"]

    predicted_class = result["predicted_class"]
    confidence = result["confidence"]

    st.markdown("---")

    st.markdown(
        '<div class="section-title">📊 Prediction Dashboard</div>',
        unsafe_allow_html=True
    )

    col1, col2, col3 = st.columns(3)

    with col1:

        st.markdown(f"""
        <div class="info-card">
            <div class="info-label">PREDICTED CLASS</div>
            <div class="info-value">{predicted_class}</div>
        </div>
        """, unsafe_allow_html=True)

    with col2:

        st.markdown(f"""
        <div class="info-card">
            <div class="info-label">MODEL CONFIDENCE</div>
            <div class="info-value">{confidence * 100:.2f}%</div>
        </div>
        """, unsafe_allow_html=True)

    with col3:

        st.markdown("""
        <div class="info-card">
            <div class="info-label">MODEL ARCHITECTURE</div>
            <div class="info-value">MangoLite-CNN v2</div>
        </div>
        """, unsafe_allow_html=True)

    st.write("")

    st.progress(
        float(np.clip(confidence, 0, 1)),
        text=f"Prediction confidence: {confidence * 100:.2f}%"
    )

    st.caption(
        "Confidence represents the model's predicted probability "
        "for the selected class. It is not a guarantee of diagnosis."
    )


    # =====================================================
    # 8. CLASS PROBABILITY CHART
    # =====================================================

    st.markdown("---")

    st.markdown(
        '<div class="section-title">📈 Disease Probability Distribution</div>',
        unsafe_allow_html=True
    )

    probabilities = result["probabilities"]

    sorted_items = sorted(
        probabilities.items(),
        key=lambda x: x[1]
    )

    labels = [item[0] for item in sorted_items]
    values = [item[1] * 100 for item in sorted_items]

    fig, ax = plt.subplots(figsize=(11, 5))

    bars = ax.barh(
        labels,
        values,
        color="#397A50",
        height=0.62
    )

    for bar, value in zip(bars, values):

        ax.text(
            value + 0.8,
            bar.get_y() + bar.get_height() / 2,
            f"{value:.2f}%",
            va="center",
            fontsize=9,
            color="#174D35",
            fontweight="bold"
        )

    ax.set_xlim(0, 115)

    ax.set_xlabel(
        "Prediction Probability (%)",
        color="#174D35",
        fontsize=10
    )

    ax.set_title(
        "Probability Across Eight Mango Leaf Classes",
        color="#174D35",
        fontsize=13,
        fontweight="bold",
        pad=15
    )

    ax.set_facecolor("#FFFFFF")
    fig.patch.set_facecolor("#FFFFFF")

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)

    ax.spines["bottom"].set_color("#DCE6DA")

    ax.tick_params(axis="y", colors="#174D35", labelsize=9)
    ax.tick_params(axis="x", colors="#52715C")

    ax.grid(
        axis="x",
        linestyle="--",
        alpha=0.2
    )

    ax.set_axisbelow(True)

    plt.tight_layout()

    st.pyplot(fig, use_container_width=True)

    plt.close(fig)


    # =====================================================
    # 9. GRAD-CAM EXPLAINABILITY
    # =====================================================

    st.markdown("---")

    st.markdown(
        '<div class="section-title">🔬 Explainable AI — Grad-CAM</div>',
        unsafe_allow_html=True
    )

    st.write(
        "Grad-CAM highlights image regions that contribute "
        "to the model's prediction."
    )

    tab1, tab2 = st.tabs([
        "Visual Explanation",
        "About Grad-CAM"
    ])

    with tab1:

        fig, axes = plt.subplots(
            1, 3,
            figsize=(16, 5)
        )

        resized_image = original_image.resize((224, 224))

        axes[0].imshow(resized_image)
        axes[0].set_title(
            "Original Leaf",
            fontsize=12,
            fontweight="bold",
            color="#174D35"
        )

        axes[1].imshow(heatmap, cmap="jet")
        axes[1].set_title(
            "Grad-CAM Heatmap",
            fontsize=12,
            fontweight="bold",
            color="#174D35"
        )

        axes[2].imshow(resized_image)
        axes[2].imshow(
            heatmap,
            cmap="jet",
            alpha=0.4
        )

        axes[2].set_title(
            "Grad-CAM Overlay",
            fontsize=12,
            fontweight="bold",
            color="#174D35"
        )

        for ax in axes:
            ax.axis("off")

        plt.tight_layout()

        st.pyplot(fig, use_container_width=True)

        plt.close(fig)

    with tab2:

        st.markdown("""
        **Gradient-weighted Class Activation Mapping (Grad-CAM)**

        Grad-CAM is an explainable AI technique that uses
        gradients flowing into a convolutional layer to
        generate a visual explanation of a model prediction.

        - **Original Leaf:** The uploaded input image.
        - **Heatmap:** Highlights regions with stronger
          influence on the selected class prediction.
        - **Overlay:** Combines the heatmap with the original
          image for visual interpretation.

        **Research note:** Grad-CAM provides an approximate
        visual explanation. Highlighted regions should not
        be interpreted as verified disease boundaries.
        """)


# =========================================================
# 10. FOOTER
# =========================================================

st.markdown("""
<div class="footer">
    <strong>🥭 MangoVision AI</strong>
    <br><br>
    Powered by MangoLite-CNN v2
    <br>
    Lightweight Deep Learning | Explainable AI
    <br><br>
    Research Prototype · 2026
</div>
""", unsafe_allow_html=True)
