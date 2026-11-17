
import streamlit as st
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

from src.prediction import load_model, preprocess_image, predict_image
from src.gradcam import generate_gradcam


# ============================================================
# 1. PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="MangoVision AI",
    page_icon="🥭",
    layout="wide",
    initial_sidebar_state="expanded"
)

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = (
    BASE_DIR / "models" / "MangoLite_CNN_v2_100K_Best.keras"
)


# ============================================================
# 2. PROFESSIONAL GREEN THEME + HIGH CONTRAST
# ============================================================

st.markdown("""
<style>

@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700;800&display=swap');

:root {
    --forest: #123D2A;
    --green: #174D35;
    --botanical: #286344;
    --sage: #A8C99B;
    --pale: #EAF2E8;
    --cream: #F7F6EF;
    --white: #FFFFFF;
    --text: #183D2B;
    --muted: #496651;
}

/* Global */
html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    color: var(--text) !important;
}

.stApp {
    background: var(--cream);
}

.block-container {
    max-width: 1450px;
    padding-top: 2rem;
    padding-bottom: 3rem;
}

/* Main text */
[data-testid="stAppViewContainer"] p,
[data-testid="stAppViewContainer"] li,
[data-testid="stAppViewContainer"] label {
    color: var(--text);
}

[data-testid="stMarkdownContainer"] p,
[data-testid="stMarkdownContainer"] li {
    color: var(--text) !important;
}

/* Headings */
h1, h2, h3, h4, h5, h6 {
    color: var(--green) !important;
    font-weight: 800 !important;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: var(--forest);
}

[data-testid="stSidebar"] * {
    color: #FFFFFF !important;
}

[data-testid="stSidebar"] p {
    color: #E2EEE3 !important;
}

/* Hero */
.hero {
    background: linear-gradient(120deg, #123D2A, #286344);
    padding: 38px;
    border-radius: 20px;
    margin-bottom: 28px;
    box-shadow: 0 8px 25px rgba(18, 61, 42, 0.12);
}

.hero h1 {
    color: #FFFFFF !important;
    font-size: 42px;
    font-weight: 800;
    margin: 0 0 8px 0;
}

.hero p {
    color: #FFFFFF !important;
    font-size: 16px;
    margin-bottom: 5px;
}

/* Section titles */
.section-title {
    color: #174D35 !important;
    font-size: 24px;
    font-weight: 800;
    margin-top: 16px;
    margin-bottom: 14px;
}

/* Information cards */
.info-card {
    background: #FFFFFF;
    border: 1px solid #DCE7DB;
    border-radius: 16px;
    padding: 23px;
    min-height: 125px;
    box-shadow: 0 4px 15px rgba(18, 61, 42, 0.055);
}

.info-label {
    color: #496651 !important;
    font-size: 12px;
    font-weight: 800;
    letter-spacing: 1px;
}

.info-value {
    color: #123D2A !important;
    font-size: 26px;
    font-weight: 800;
    margin-top: 10px;
    overflow-wrap: anywhere;
}

/* Upload area */
[data-testid="stFileUploader"] {
    background: #EAF2E8;
    border: 2px dashed #83A98A;
    border-radius: 16px;
    padding: 15px;
}

[data-testid="stFileUploader"] * {
    color: #183D2B !important;
}

/* Buttons */
.stButton > button {
    background: #174D35;
    color: #FFFFFF !important;
    border: 1px solid #174D35;
    border-radius: 11px;
    padding: 12px 22px;
    font-weight: 700;
    min-height: 48px;
}

.stButton > button:hover {
    background: #286344;
    color: #FFFFFF !important;
    border-color: #286344;
}

/* Tabs */
.stTabs [data-baseweb="tab"] {
    color: #174D35 !important;
    font-weight: 700;
}

.stTabs [aria-selected="true"] {
    color: #174D35 !important;
}

/* Captions */
[data-testid="stCaptionContainer"] p {
    color: #496651 !important;
}

/* Footer */
.footer {
    background: #123D2A;
    color: #FFFFFF !important;
    padding: 24px;
    border-radius: 16px;
    text-align: center;
    margin-top: 38px;
}

.footer * {
    color: #FFFFFF !important;
}

/* Alerts */
[data-testid="stAlert"] {
    border-radius: 12px;
}

/* Dividers */
hr {
    border-color: #DCE7DB !important;
}

</style>
""", unsafe_allow_html=True)


# ============================================================
# 3. MODEL LOADING
# ============================================================

@st.cache_resource
def get_model():
    return load_model(str(MODEL_PATH))


# ============================================================
# 4. SIDEBAR
# ============================================================

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

    st.write("Input resolution: 224 × 224")

    st.write("Classification classes: 8")

    st.write("Explainability: Grad-CAM")

    st.markdown("---")

    st.markdown("### 🔬 Research Focus")

    st.write("• Lightweight deep learning")

    st.write("• Disease classification")

    st.write("• Explainable artificial intelligence")

    st.write("• Efficient inference")

    st.markdown("---")

    st.caption("Research Prototype | 2026")


# ============================================================
# 5. HERO HEADER
# ============================================================

st.markdown("""
<div class="hero">
    <h1>🥭 MangoVision AI</h1>

    <p style="font-size:20px;font-weight:700;">
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


# ============================================================
# 6. IMAGE UPLOAD SECTION
# ============================================================

st.markdown(
    '<div class="section-title">🌱 Leaf Analysis Studio</div>',
    unsafe_allow_html=True
)

st.markdown(
    "Upload a mango leaf image to identify its condition "
    "and explore the model's visual explanation."
)

uploaded_file = st.file_uploader(
    "Choose a mango leaf image",
    type=["jpg", "jpeg", "png"],
    help="Supported image formats: JPG, JPEG and PNG"
)


# Clear old analysis when a different image is uploaded

if uploaded_file is not None:

    image_bytes = uploaded_file.getvalue()

    if st.session_state.get("image_bytes") != image_bytes:

        st.session_state.pop("analysis_result", None)

        st.session_state["image_bytes"] = image_bytes


# ============================================================
# 7. IMAGE PREVIEW AND ANALYSIS
# ============================================================

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    preview_col, action_col = st.columns([1, 1])

    with preview_col:

        st.markdown("#### Uploaded Mango Leaf")

        st.image(
            image,
            use_container_width=True
        )

    with action_col:

        st.markdown("#### Ready for Analysis?")

        st.markdown("""
        <div class="info-card">
            <div class="info-label">IMAGE STATUS</div>
            <div class="info-value">Image Loaded ✓</div>
            <p style="color:#294936 !important;">
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

                st.success(
                    "Leaf analysis completed successfully!"
                )

            except Exception as e:

                st.error("Analysis could not be completed.")

                st.exception(e)


# ============================================================
# 8. PREDICTION DASHBOARD
# ============================================================

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


    # ========================================================
    # 9. DISEASE PROBABILITY DISTRIBUTION
    # ========================================================

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

    ax.tick_params(
        axis="y",
        colors="#174D35",
        labelsize=9
    )

    ax.tick_params(
        axis="x",
        colors="#52715C"
    )

    ax.grid(
        axis="x",
        linestyle="--",
        alpha=0.2
    )

    ax.set_axisbelow(True)

    plt.tight_layout()

    st.pyplot(
        fig,
        use_container_width=True
    )

    plt.close(fig)


    # ========================================================
    # 10. EXPLAINABLE AI - GRAD-CAM
    # ========================================================

    st.markdown("---")

    st.markdown(
        '<div class="section-title">🔬 Explainable AI — Grad-CAM</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        "Grad-CAM highlights image regions that contribute "
        "to the model's prediction."
    )

    tab1, tab2 = st.tabs([
        "Visual Explanation",
        "About Grad-CAM"
    ])

    with tab1:

        fig, axes = plt.subplots(
            1,
            3,
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

        axes[1].imshow(
            heatmap,
            cmap="jet"
        )

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

        st.pyplot(
            fig,
            use_container_width=True
        )

        plt.close(fig)

    with tab2:

        st.markdown("""
        <div class="info-card">

        <h3 style="color:#174D35;">
        Gradient-weighted Class Activation Mapping (Grad-CAM)
        </h3>

        <p style="color:#183D2B;">
        Grad-CAM is an explainable AI technique that uses
        gradients flowing into a convolutional layer to
        generate a visual explanation of a model prediction.
        </p>

        <p style="color:#183D2B;">
        <b>Original Leaf:</b> The uploaded input image.
        </p>

        <p style="color:#183D2B;">
        <b>Heatmap:</b> Highlights regions with stronger
        influence on the selected class prediction.
        </p>

        <p style="color:#183D2B;">
        <b>Overlay:</b> Combines the heatmap with the original
        image for visual interpretation.
        </p>

        <hr>

        <p style="color:#294936;">
        <b>Research note:</b> Grad-CAM provides an approximate
        visual explanation. Highlighted regions should not
        be interpreted as verified disease boundaries.
        </p>

        </div>
        """, unsafe_allow_html=True)


# ============================================================
# 11. FOOTER
# ============================================================

st.markdown("""
<div class="footer">

    <strong style="font-size:20px;">
        🥭 MangoVision AI
    </strong>

    <br><br>

    Powered by MangoLite-CNN v2

    <br>

    Lightweight Deep Learning | Explainable AI

    <br><br>

    Research Prototype · 2026

</div>
""", unsafe_allow_html=True)
