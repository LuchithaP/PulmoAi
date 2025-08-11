import torch
from PIL import Image
from model import CovidCNN
from preprocess import transform


def predict_image(image_path, model, transform, device):
    model.eval()
    try:
        img = Image.open(image_path).convert("RGB")
        img = transform(img).unsqueeze(0).to(device)
        with torch.no_grad():
            output = model(img)
            prediction = output.item()
            label = 1 if prediction >= 0.8 else 0
        return "Positive" if label == 1 else "Negative", prediction
    except Exception as e:
        print(f"Error predicting {image_path}: {e}")
        return None, None


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CovidCNN().to(device)
    model.load_state_dict(torch.load("Enter your model.pth", map_location=device))
    image_path = "Enter path to image for prediction"
    result, confidence = predict_image(image_path, model, transform, device)
    if result:
        print(f"Covid Prediction: {result} (Confidence: {confidence:.4f})")
