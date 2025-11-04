import gradio as gr
import torch
import torch.nn.functional as F
from PIL import Image
import json
from torchvision import transforms
from model import resnet50

# Load class labels from local file
with open("imagenet_classes.json", "r") as f:
    class_labels = json.load(f)

# Load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = resnet50(num_classes=1000, drop_path_rate=0.0, use_blurpool=True)
# model.load_state_dict(torch.load("best_resnet50_imagenet_1k.pt", map_location=device))
checkpoint = torch.load('best_resnet50_imagenet_1k.pt', map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
model.to(device)
model.eval()

# Image preprocessing
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def predict(image):
    # Preprocess
    img_tensor = transform(image).unsqueeze(0).to(device)
    
    # Predict
    with torch.no_grad():
        outputs = model(img_tensor)
        probabilities = F.softmax(outputs, dim=1)[0]
    
    # Get top 5 predictions
    top5_prob, top5_idx = torch.topk(probabilities, 5)
    
    # Format results
    results = {class_labels[idx]: float(prob) for idx, prob in zip(top5_idx, top5_prob)}
    return results

# Create Gradio interface
demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil"),
    outputs=gr.Label(num_top_classes=5),
    title="ImageNet ResNet50 Classifier (71% Accuracy)",
    description="ResNet50 trained on ImageNet with improved stem, BlurPool, and progressive resizing. Achieved 71% top-1 accuracy under $30 budget.",
    examples=[]
)

if __name__ == "__main__":
    demo.launch()
