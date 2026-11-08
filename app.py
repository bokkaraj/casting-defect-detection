import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

@st.cache_resource
def load_model():
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 2)
    model.load_state_dict(torch.load("best_resnet18_small.pt", map_location="cpu"))
    model.eval()
    return model

model = load_model()

# Same preprocessing as training (no augmentation)
tf = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

st.title("Casting Defect Detector")
st.write(
    "ResNet18 classifier trained on cast impeller images. "
    "Test set: 99.3% defect recall, 100% defect precision. "
    "Not for production use. Always verify flagged parts."
)

file = st.file_uploader("Upload a casting photo", type=["jpeg", "jpg", "png"])

if file is not None:
    img = Image.open(file)
    st.image(img, width=300)
    x = tf(img).unsqueeze(0)
    with torch.no_grad():
        probs = torch.softmax(model(x), dim=1)[0]
    # Class 0 = def_front (defective), class 1 = ok_front (OK)
    p_defect = float(probs[0])
    if p_defect >= 0.5:
        st.error(f"DEFECTIVE: send for inspection ({p_defect:.1%} defect probability)")
    else:
        st.success(f"OK ({1 - p_defect:.1%} confidence)")
