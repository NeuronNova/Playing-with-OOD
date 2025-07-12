import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
import pickle
from streamlit_drawable_canvas import st_canvas

# CONFIG
st.set_page_config(page_title="Digit Classifier", layout="wide")


# Utility functions
def cosine_distance(a, b):
    """Calculate cosine distance between two vectors"""
    dot_product = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 1.0
    cosine_sim = dot_product / (norm_a * norm_b)
    return 1 - cosine_sim


def beta_certainty(digit, alpha_count, beta_count):
    """Calculate beta distribution statistics"""
    alpha = alpha_count.get(digit, 1)
    beta = beta_count.get(digit, 1)
    mean = alpha / (alpha + beta)
    variance = (alpha * beta) / ((alpha + beta) ** 2 * (alpha + beta + 1))

    if mean > 0 and mean < 1:
        entropy = -mean * np.log(mean) - (1 - mean) * np.log(1 - mean)
    else:
        entropy = 0

    return mean, variance, entropy


# Load model and data
@st.cache_resource
def load_model_and_data():
    try:
        model = tf.keras.models.load_model("cnn_model.h5")
        embedding_model = tf.keras.Model(inputs=model.input, outputs=model.get_layer(index=-2).output)
        prototypes = np.load("prototypes.npy", allow_pickle=True).item()

        with open("alpha_count.pkl", "rb") as f:
            alpha_count = pickle.load(f)
        with open("beta_count.pkl", "rb") as f:
            beta_count = pickle.load(f)

        return model, embedding_model, prototypes, alpha_count, beta_count, None
    except Exception as e:
        return None, None, None, None, None, str(e)


# Load everything
model, embedding_model, prototypes, alpha_count, beta_count, error = load_model_and_data()

st.title("🔢 Digit Classifier")

if error:
    st.error(f"Error: {error}")
    st.stop()

# Fixed thresholds
dist_threshold = 0.25
var_threshold = 0.03
confidence_threshold = 0.1

# Canvas
col1, col2 = st.columns([1, 1])

with col1:
    st.header("✏️ Draw a Digit")
    canvas_result = st_canvas(
        fill_color="white",
        stroke_width=20,
        stroke_color="white",
        background_color="black",
        height=400,
        width=400,
        drawing_mode="freedraw",
        key="canvas",
    )

with col2:
    st.header("📊 Results")

    if canvas_result.image_data is not None:
        try:
            # Process canvas
            img_data = canvas_result.image_data[:, :, 0]
            img_data = 255 - img_data
            img = Image.fromarray(img_data.astype("uint8")).resize((28, 28))
            img_array = np.array(img).astype("float32") / 255.0
            x = img_array.reshape(1, 28, 28, 1)

            # Show processed image
            st.image(img, caption="Processed Image", width=100)

            # Get predictions
            probs = model.predict(x, verbose=0)
            pred = int(np.argmax(probs))
            confidence = float(probs[0][pred])

            # Get embedding and calculate distance
            emb = embedding_model.predict(x, verbose=0)[0]
            dist = cosine_distance(emb, prototypes[pred])

            # Calculate beta certainty
            mean, var, ent = beta_certainty(pred, alpha_count, beta_count)

            # Decision logic - matching your Colab logic
            reject_reasons = []
            if var > var_threshold:
                reject_reasons.append("High uncertainty (Beta var)")
            if dist > dist_threshold:
                reject_reasons.append("Too far from prototype")

            is_accepted = len(reject_reasons) == 0

            # Results
            if is_accepted:
                st.success(f"✅ ACCEPTED")
                st.metric("Predicted Digit", pred)
            else:
                st.error(f"❌ REJECTED")
                st.write("**Reasons:**")
                for reason in reject_reasons:
                    st.write(f"• {reason}")

            # Show key metrics
            st.write("---")
            st.write(f"**Trust:** {mean:.2f}")
            st.write(f"**Variance:** {var:.4f} (threshold: {var_threshold})")
            st.write(f"**Distance:** {dist:.3f} (threshold: {dist_threshold})")
            st.write(f"**Entropy:** {ent:.4f}")
            st.write(f"**Confidence:** {confidence:.3f}")

        except Exception as e:
            st.error(f"Error: {str(e)}")
    else:
        st.info("👆 Draw a digit to get started")
