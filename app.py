import pickle
import numpy as np
import cv2
import gradio as gr

# ── Load Model ───────────────────────────────────────────
with open("model/model.pkl", "rb") as f:
    bundle = pickle.load(f)

clf = bundle["clf"]
pca = bundle["pca"]

IMG_SIZE = (62, 47)  # LFW dataset default size

# ── Predict Function ─────────────────────────────────────
def predict(image):
    if image is None:
        return {"Arnold": 0.0, "Others": 0.0}

    # Convert to grayscale and resize to match training data
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    resized = cv2.resize(gray, IMG_SIZE)
    flattened = resized.flatten().reshape(1, -1)

    # Apply same PCA transformation used in training
    pca_features = pca.transform(flattened)

    # Get probabilities
    probs = clf.predict_proba(pca_features)[0]
    labels = clf.classes_

    result = {label: float(prob) for label, prob in zip(labels, probs)}
    return result

# ── Gradio UI ─────────────────────────────────────────────
with gr.Blocks(theme=gr.themes.Soft(), title="Arnold Facial Recognition") as demo:
    gr.Markdown("""
    # 🎭 Facial Recognition: Arnold vs Others
    ### Upload a face image to identify if it's Arnold Schwarzenegger!
    """)

    with gr.Row():
        with gr.Column():
            image_input = gr.Image(type="numpy", label="📸 Upload Face Image")
            predict_btn = gr.Button("🔍 Identify", variant="primary")

        with gr.Column():
            output_label = gr.Label(num_top_classes=2, label="🧠 Prediction")
            gr.Markdown("""
            **How it works:**
            - Model trained on LFW dataset
            - Uses PCA (Eigenfaces) + SVM
            - 94% accuracy
            """)

    predict_btn.click(fn=predict, inputs=image_input, outputs=output_label)
    image_input.change(fn=predict, inputs=image_input, outputs=output_label)

    gr.Markdown("*Model trained using scikit-learn on Labeled Faces in the Wild dataset*")

if __name__ == "__main__":
    demo.launch()
