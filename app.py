import streamlit as st
import numpy as np
from PIL import Image
import onnxruntime as ort
import shap
import matplotlib.pyplot as plt


# ============================================================
# Page configuration
# ============================================================

st.set_page_config(
    page_title="Eye Disease Classifier",
    page_icon="👁️",
    layout="wide"
)


# ============================================================
# Load ONNX model
# ============================================================

@st.cache_resource
def load_model():
    return ort.InferenceSession(
        "model/model.onnx"
    )


model = load_model()


# ============================================================
# Model input/output
# ============================================================

INPUT_NAME = model.get_inputs()[0].name
OUTPUT_NAME = model.get_outputs()[0].name


# ============================================================
# Classes
# ============================================================

CLASS_NAMES = [
    "Cataract",
    "Diabetic Retinopathy",
    "Glaucoma",
    "Normal"
]


# ============================================================
# ONNX prediction function
# ============================================================

def predict(images):

    images = np.asarray(
        images,
        dtype=np.float32
    )

    outputs = model.run(
        [OUTPUT_NAME],
        {
            INPUT_NAME: images
        }
    )[0]

    return outputs


# ============================================================
# Convert model output to probabilities
# ============================================================

def get_probabilities(output):

    output = np.asarray(
        output,
        dtype=np.float32
    )

    # If model already contains Softmax
    if np.all(output >= 0) and np.isclose(
        np.sum(output),
        1.0,
        atol=1e-3
    ):
        return output

    # Otherwise output is logits
    exp_output = np.exp(
        output - np.max(output)
    )

    return exp_output / np.sum(exp_output)


# ============================================================
# SHAP prediction function
# ============================================================

def shap_predict(images):

    predictions = predict(images)

    probabilities = []

    for prediction in predictions:
        probabilities.append(
            get_probabilities(prediction)
        )

    return np.asarray(probabilities)


# ============================================================
# Create SHAP explainer
# ============================================================

@st.cache_resource
def create_shap_explainer():

    # Image shape expected by ONNX model
    image_shape = (224, 224, 3)

    masker = shap.maskers.Image(
        "blur(32,32)",
        image_shape
    )

    explainer = shap.Explainer(
        shap_predict,
        masker,
        output_names=CLASS_NAMES
    )

    return explainer


# ============================================================
# Title
# ============================================================

st.title("👁️ Eye Disease Classification")

st.write(
    "Upload a retinal image and the AI model will "
    "predict the most likely eye condition."
)

st.warning(
    "⚠️ This AI model can make mistakes and is not a "
    "medical diagnosis. Please consult a qualified "
    "medical professional for diagnosis and treatment."
)


# ============================================================
# Upload
# ============================================================

uploaded_file = st.file_uploader(
    "Upload a retinal image",
    type=["jpg", "jpeg", "png"]
)


# ============================================================
# Main prediction
# ============================================================

if uploaded_file is not None:

    # --------------------------------------------------------
    # Load image
    # --------------------------------------------------------

    image = Image.open(
        uploaded_file
    ).convert("RGB")


    # --------------------------------------------------------
    # Resize
    # --------------------------------------------------------

    image_resized = image.resize(
        (224, 224)
    )


    # --------------------------------------------------------
    # Convert to NumPy
    # --------------------------------------------------------

    image_array = np.array(
        image_resized
    ).astype(np.float32)


    # Add batch dimension
    image_array = np.expand_dims(
        image_array,
        axis=0
    )


    # ========================================================
    # Run ONNX model
    # ========================================================

    raw_prediction = predict(
        image_array
    )[0]

    prediction = get_probabilities(
        raw_prediction
    )


    # ========================================================
    # Prediction
    # ========================================================

    predicted_index = np.argmax(
        prediction
    )

    predicted_class = CLASS_NAMES[
        predicted_index
    ]

    confidence = (
        prediction[predicted_index] * 100
    )


    # ========================================================
    # Display uploaded image
    # ========================================================

    st.divider()

    st.subheader("Uploaded Retinal Image")

    st.image(
        image,
        width=450
    )


    # ========================================================
    # Prediction result
    # ========================================================

    st.divider()

    st.subheader("AI Prediction")

    col1, col2 = st.columns(2)

    with col1:

        st.metric(
            "Predicted Condition",
            predicted_class
        )

    with col2:

        st.metric(
            "Confidence",
            f"{confidence:.2f}%"
        )


    # ========================================================
    # Confidence check
    # ========================================================

    if confidence < 60:

        st.error(
            "⚠️ The model is not sufficiently confident "
            "in this prediction."
        )

        st.warning(
            "Please consult a qualified doctor or "
            "ophthalmologist for a professional evaluation."
        )

    else:

        st.success(
            f"The model predicts **{predicted_class}** "
            f"with {confidence:.2f}% confidence."
        )


    # ========================================================
    # Probability distribution
    # ========================================================

    st.subheader(
        "Prediction Probabilities"
    )

    for class_name, probability in zip(
        CLASS_NAMES,
        prediction
    ):

        percentage = probability * 100

        st.write(
            f"**{class_name}** — "
            f"{percentage:.2f}%"
        )

        st.progress(
            float(probability)
        )


    # ========================================================
    # SHAP explanation
    # ========================================================

    st.divider()

    st.subheader(
        "🔍 Model Explanation"
    )

    st.write(
        "The heatmap shows which areas of the retinal image "
        "influenced the model's prediction."
    )


    # Only generate SHAP when prediction is reasonably confident
    if confidence >= 60:

        with st.spinner(
            "Generating AI explanation..."
        ):

            explainer = create_shap_explainer()

            shap_values = explainer(
                image_array,
                max_evals=1000,
                batch_size=32,
                outputs=shap.Explanation.argsort.flip[:1]
            )


        # ----------------------------------------------------
        # Prepare image for visualization
        # ----------------------------------------------------

        image_display = np.clip(
            image_array[0] / 255.0,
            0,
            1
        )


        # ----------------------------------------------------
        # Extract SHAP values
        # ----------------------------------------------------

        values = shap_values.values[0]


        # Remove output dimension
        if values.ndim == 4:
            values = values[..., 0]


        # Combine RGB channels
        shap_map = np.sum(
            values,
            axis=-1
        )


        # ----------------------------------------------------
        # Improve heatmap visibility
        # ----------------------------------------------------

        limit = np.percentile(
            np.abs(shap_map),
            98
        )

        # Prevent zero division
        if limit == 0:
            limit = 1e-8

        shap_map = np.clip(
            shap_map,
            -limit,
            limit
        )


        # ====================================================
        # Create clean visualization
        # ====================================================

        fig = plt.figure(
            figsize=(13, 6)
        )

        grid = fig.add_gridspec(
            1,
            3,
            width_ratios=[1, 1, 0.07],
            wspace=0.15
        )


        # ----------------------------------------------------
        # LEFT: Original image
        # ----------------------------------------------------

        ax_original = fig.add_subplot(
            grid[0, 0]
        )

        ax_original.imshow(
            image_display
        )

        ax_original.set_title(
            "Input Image",
            fontsize=16,
            fontweight="bold",
            pad=12
        )

        ax_original.axis("off")


        # ----------------------------------------------------
        # RIGHT: SHAP explanation
        # ----------------------------------------------------

        ax_shap = fig.add_subplot(
            grid[0, 1]
        )

        ax_shap.imshow(
            image_display
        )

        heatmap = ax_shap.imshow(
            shap_map,
            cmap="RdBu_r",
            alpha=0.65,
            vmin=-limit,
            vmax=limit
        )

        ax_shap.set_title(
            "Areas Influencing the Prediction",
            fontsize=16,
            fontweight="bold",
            pad=12
        )

        ax_shap.axis("off")


        # ----------------------------------------------------
        # COLORBAR — separate column
        # ----------------------------------------------------

        ax_colorbar = fig.add_subplot(
            grid[0, 2]
        )

        colorbar = fig.colorbar(
            heatmap,
            cax=ax_colorbar
        )

        colorbar.set_label(
            "Influence on Prediction",
            fontsize=11,
            labelpad=10
        )


        # ----------------------------------------------------
        # Main title
        # ----------------------------------------------------

        fig.suptitle(
            f"AI Prediction: "
            f"{predicted_class} "
            f"({confidence:.1f}% confidence)",
            fontsize=20,
            fontweight="bold",
            y=0.98
        )


        # ----------------------------------------------------
        # Explanation legend
        # ----------------------------------------------------

        fig.text(
            0.5,
            0.02,
            "🔴 Red areas support the prediction    "
            "🔵 Blue areas work against the prediction",
            ha="center",
            fontsize=11
        )


        # ----------------------------------------------------
        # Layout
        # ----------------------------------------------------

        fig.subplots_adjust(
            top=0.82,
            bottom=0.12,
            left=0.03,
            right=0.96
        )


        # ----------------------------------------------------
        # Display in Streamlit
        # ----------------------------------------------------

        st.pyplot(
            fig,
            use_container_width=True
        )

        plt.close(fig)


    else:

        st.info(
            "The model explanation is not displayed because "
            "the prediction confidence is below 60%. "
            "Please seek professional medical advice."
        )