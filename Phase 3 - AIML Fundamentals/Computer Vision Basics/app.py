import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import os

# ── Page config ───────────────────────────────────────────
st.set_page_config(
    page_title="Cat vs Dog Classifier",
    page_icon="🐾",
    layout="centered"
)

# ── Load model (cached so it loads only once) ─────────────
@st.cache_resource
def load_model():
    ckpt_path = "C:/Users/hp/cnn_project/resnet50_cat_dog.pth"
    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt      = torch.load(ckpt_path, map_location=device)

    model = models.resnet50(weights=None)
    model.fc = nn.Sequential(
        nn.Dropout(p=0.4),
        nn.Linear(2048, 2)
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device).eval()
    return model, ckpt["class_names"], device

model, class_names, device = load_model()

# ── Transform ─────────────────────────────────────────────
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])

# ── Predict function ──────────────────────────────────────
def predict(img: Image.Image):
    tensor = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        probs = torch.softmax(model(tensor), dim=1)[0]
    return {class_names[i]: probs[i].item() for i in range(len(class_names))}

# ── UI ────────────────────────────────────────────────────
st.title("🐾 Cat vs Dog Classifier")
st.markdown("Upload any image and the ResNet50 model will classify it.")
st.divider()

uploaded = st.file_uploader(
    "Choose an image", type=["jpg", "jpeg", "png"]
)

if uploaded:
    img = Image.open(uploaded).convert("RGB")

    col1, col2 = st.columns(2)

    with col1:
        st.image(img, caption="Uploaded Image", use_container_width=True)

    with col2:
        with st.spinner("Classifying..."):
            results = predict(img)

        pred  = max(results, key=results.get)
        conf  = results[pred] * 100
        other = [k for k in results if k != pred][0]

        # Emoji per class
        emoji = "🐱" if pred == "cat" else "🐶"

        st.markdown(f"### {emoji} Prediction: `{pred.upper()}`")
        st.metric("Confidence", f"{conf:.2f}%")
        st.divider()

        # Confidence bars
        st.markdown("**Class probabilities**")
        for cls, prob in sorted(results.items(),
                                key=lambda x: x[1], reverse=True):
            icon = "🐱" if cls == "cat" else "🐶"
            st.markdown(f"{icon} **{cls}**")
            st.progress(prob, text=f"{prob*100:.2f}%")

        
