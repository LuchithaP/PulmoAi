import os
import torch
from flask import Flask, render_template, request
from model import CovidCNN
from preprocess import transform
from predict import predict_image

# Flask setup
app = Flask(__name__)
UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# Load model once when server starts
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CovidCNN().to(device)
model.load_state_dict(torch.load("covid_classifier_model.pth", map_location=device))
model.eval()


@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    confidence = None
    image_url = None

    if request.method == "POST":
        file = request.files.get("file")
        if file:
            filename = file.filename
            filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
            file.save(filepath)

            result, confidence = predict_image(filepath, model, transform, device)
            if confidence is not None:
                confidence = round(confidence, 4)

            image_url = f"{app.config['UPLOAD_FOLDER']}/{filename}"

    return render_template(
        "index.html", result=result, confidence=confidence, image_url=image_url
    )


if __name__ == "__main__":
    app.run(debug=True)
