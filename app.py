import os
import easyocr
from fuzzywuzzy import fuzz
import torch
import torch.nn as nn
from torchvision import models, transforms
from flask import Flask, request, jsonify, render_template_string
from PIL import Image
import warnings
from werkzeug.utils import secure_filename

# Suppress the fuzzywuzzy warning about python-Levenshtein
warnings.filterwarnings("ignore", message="Using slow pure-python SequenceMatcher")

BAD_WORDS = [
    "fuck", "shit", "ass", "bitch", "idiot", "moron", "nigger", "cunt", "whore", "slut",
    "bastard", "dick", "pussy", "damn", "hell", "crap", "douche", "fag", "retard", "screw",
    "fuck you", "son of a bitch", "motherfucker", "asshole", "dumbass", "shithead", "cock", "wanker",
    "عرص", "كس", "طيز", "زب", "كلب", "عاهر", "قحبة", "كفر", "ملحد", "شرموطة", "عاهره",
    "منيوك", "منيوكة", "زبالة", "خول", "دعارة", "فاجر", "فاسق", "فاحشة", "ممحونة", "ممحون",
    "ابن الكلب", "ابن العاهرة", "ابن الشرموطة", "يا خول", "يا عاهر", "يا كلب", "يا زبالة"
]

# Initialize OCR reader
reader = easyocr.Reader(['en', 'ar'])

# Define the model - Fixed version
class SafeImageClassifier(nn.Module):
    def __init__(self):
        super(SafeImageClassifier, self).__init__()
        # Load pretrained ResNet18 model
        self.resnet = models.resnet18(weights=None)
        # Replace the final fully connected layer
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, 2)  # 2 classes: safe/unsafe
    
    def forward(self, x):
        return self.resnet(x)

# Image transformation
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Initialize Flask app
app = Flask(__name__)

# Initialize and load the model
model = SafeImageClassifier()
state_dict = torch.load("model/resnet18_model.pth", map_location="cpu")
# The next line is the key fix - loading the state dict correctly
model.resnet.load_state_dict(state_dict, strict=False)
model.eval()

# Ensure upload directory exists
os.makedirs("uploads", exist_ok=True)

# Define allowed file extensions
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# HTML template with improved feedback display
INDEX_TEMPLATE = """
<!DOCTYPE html>
<html>
    <head>
        <title>Image Safety Analyzer</title>
        <style>
            body { font-family: Arial, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; }
            h1 { color: #333; }
            form { margin: 20px 0; padding: 20px; border: 1px solid #ddd; border-radius: 5px; }
            .result { margin-top: 20px; padding: 15px; border-radius: 5px; }
            .success { background-color: #d4edda; border: 1px solid #c3e6cb; color: #155724; }
            .error { background-color: #f8d7da; border: 1px solid #f5c6cb; color: #721c24; }
            .details { margin-top: 10px; padding: 10px; background-color: #f8f9fa; border-radius: 5px; }
            pre { white-space: pre-wrap; }
        </style>
    </head>
    <body>
        <h1>Image Safety Analyzer</h1>
        <p>Upload an image to analyze it for inappropriate content.</p>
        
        <form action="/upload" method="post" enctype="multipart/form-data">
            <input type="file" name="image" accept="image/*" required>
            <button type="submit">Analyze Image</button>
        </form>
        
        {% if result %}
            <div class="result {{ 'success' if result.status == 'success' else 'error' }}">
                <h3>{{ result.message }}</h3>
                {% if result.details %}
                    <div class="details">
                        <h4>Analysis Details:</h4>
                        <pre>{{ result.details | safe }}</pre>
                    </div>
                {% endif %}
            </div>
        {% endif %}
    </body>
</html>
"""

@app.route("/", methods=["GET"])
def index():
    return render_template_string(INDEX_TEMPLATE, result=None)

@app.route("/upload", methods=["POST"])
def upload_image():
    if 'image' not in request.files:
        return render_template_string(
            INDEX_TEMPLATE, 
            result={"status": "error", "message": "No image file provided"}
        )
    
    image_file = request.files['image']
    
    if not image_file or image_file.filename == '':
        return render_template_string(
            INDEX_TEMPLATE, 
            result={"status": "error", "message": "No image file uploaded"}
        )
    
    if not allowed_file(image_file.filename):
        return render_template_string(
            INDEX_TEMPLATE, 
            result={"status": "error", "message": "File type not allowed"}
        )
    
    try:
        # Secure the filename to prevent directory traversal attacks
        filename = secure_filename(image_file.filename)
        image_path = os.path.join("uploads", filename)
        image_file.save(image_path)
        
        # Analyze the image
        result = analyze_image(image_path)
        
        if isinstance(result, dict) and "error" in result:
            return render_template_string(
                INDEX_TEMPLATE, 
                result={"status": "error", "message": result["error"]}
            )
        
        # Check if image contains inappropriate content
        has_bad_words = result["ocr_result"]["has_bad_words"]
        is_unsafe_image = result["image_result"]["is_unsafe"]
        
        if has_bad_words or is_unsafe_image:
            details = []
            
            if has_bad_words:
                details.append(f"Text Analysis: {result['ocr_result']['message']}")
                details.append(f"Detected bad words: {', '.join(result['ocr_result']['detected_words'])}")
                details.append(f"Full detected text: \"{result['ocr_result']['full_text']}\"")
                
            if is_unsafe_image:
                details.append(f"Image Analysis: {result['image_result']['message']}")
                details.append(f"Safety confidence: {result['image_result']['confidence']:.2f}%")
            
            return render_template_string(
                INDEX_TEMPLATE, 
                result={
                    "status": "error", 
                    "message": "Image contains inappropriate content and cannot be uploaded.",
                    "details": "\n".join(details)
                }
            )
        
        # If everything is clean
        return render_template_string(
            INDEX_TEMPLATE, 
            result={
                "status": "success", 
                "message": "Image analysis completed successfully. No inappropriate content detected.",
                "details": f"Text Analysis: {result['ocr_result']['message']}\nImage Analysis: {result['image_result']['message']}"
            }
        )
        
    except Exception as e:
        return render_template_string(
            INDEX_TEMPLATE, 
            result={"status": "error", "message": f"Error processing image: {str(e)}"}
        )

def analyze_image(image_path):
    try:
        # Text detection with detailed results
        results = reader.readtext(image_path)
        text = " ".join([res[1] for res in results])
        
        # Find bad words with match ratio
        bad_words_found = []
        for word in BAD_WORDS:
            match_ratio = fuzz.partial_ratio(word.lower(), text.lower())
            if match_ratio > 80:
                bad_words_found.append(f"{word} ({match_ratio}%)")
        
        ocr_result = {
            "has_bad_words": len(bad_words_found) > 0,
            "message": "✅ No bad words found." if not bad_words_found else f"🚫 Bad words detected",
            "detected_words": bad_words_found,
            "full_text": text
        }
        
        # Image classification with confidence score
        image = Image.open(image_path).convert("RGB")
        input_tensor = transform(image).unsqueeze(0)
        
        with torch.no_grad():
            output = model(input_tensor)
            probabilities = torch.nn.functional.softmax(output, dim=1)
            prediction = torch.argmax(probabilities, dim=1).item()
            confidence = probabilities[0][prediction].item() * 100
        
        image_result = {
            "is_unsafe": prediction == 1,
            "message": "✅ Image is visually clean." if prediction == 0 else "🚫 Unsafe image detected.",
            "confidence": confidence,
            "prediction": prediction
        }
        
        result = {
            "ocr_result": ocr_result,
            "image_result": image_result
        }
        
        return result
    
    except Exception as e:
        return {"error": f"Error processing image: {str(e)}"}

if __name__ == "__main__":
    app.run(debug=True)