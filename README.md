# Brain-tumor-detection
# MODEL CODE
#dataset code
import kagglehub

# Download latest version
path = kagglehub.dataset_download("masoudnickparvar/brain-tumour-mri-dataset")

print("Path to dataset files:", path)

#model training
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
import joblib
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import seaborn as sns

# Step 1: Define Dataset Path
dataset_path = "/root/.cache/kagglehub/datasets/masoudnickparvar/brain-tumour-mri-dataset/versions/1"
train_dir = os.path.join(dataset_path, "Training")
test_dir = os.path.join(dataset_path, "Testing")

# Step 2: Load Data Using ImageDataGenerator
data_gen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

train_data = data_gen.flow_from_directory(train_dir, target_size=(224, 224), batch_size=32, class_mode='binary', subset='training', shuffle=False)
val_data = data_gen.flow_from_directory(train_dir, target_size=(224, 224), batch_size=32, class_mode='binary', subset='validation', shuffle=False)
test_data = data_gen.flow_from_directory(test_dir, target_size=(224, 224), batch_size=32, class_mode='binary', shuffle=False)

print(" Data Loaded Successfully!")

# Step 3: Load Pre-trained MobileNetV2 and Extract Features
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
feature_extractor=Model(inputs=base_model.input, outputs=GlobalAveragePooling2D()(base_model.output))

def extract_features(data_generator):
    features, labels = [], []
    for batch_images, batch_labels in data_generator:
        batch_features = feature_extractor.predict(batch_images)
        features.append(batch_features)
        labels.append(batch_labels)
        if len(features) * data_generator.batch_size >= data_generator.samples:
            break  # Stop when all images are processed
    return np.vstack(features), np.hstack(labels)

# Extract Features for Training, Validation, and Testing
X_train, y_train = extract_features(train_data)
X_val, y_val = extract_features(val_data)
X_test, y_test = extract_features(test_data)

print(f" Features Extracted: Train={X_train.shape}, Val={X_val.shape}, Test={X_test.shape}")

# Step 4: Apply PCA for Dimensionality Reduction
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

pca = PCA(n_components=0.95)  # Retain 95% variance
X_train_pca = pca.fit_transform(X_train_scaled)
X_val_pca = pca.transform(X_val_scaled)
X_test_pca = pca.transform(X_test_scaled)

print(f" PCA Reduced Dimensions: {X_train_pca.shape}")

# Step 5: Train Logistic Regression Classifier
print(" Training Logistic Regression Classifier...")
log_reg_classifier = LogisticRegression(max_iter=1000)
log_reg_classifier.fit(X_train_pca, y_train)

# Step 6: Predict and Evaluate Model
y_pred = log_reg_classifier.predict(X_test_pca)

# Accuracy
accuracy = accuracy_score(y_test, y_pred) * 100

# F1 Score, Precision, and Recall for Multiclass Classification
f1 = f1_score(y_test, y_pred, average='macro') * 100
precision = precision_score(y_test, y_pred, average='macro') * 100
recall = recall_score(y_test, y_pred, average='macro') * 100

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)

# Print evaluation metrics
print(f"Model Evaluation:")
print(f" Accuracy: {accuracy:.2f}%")
print(f" F1 Score (Macro Average): {f1:.2f}%")
print(f" Precision (Macro Average): {precision:.2f}%")
print(f" Recall (Macro Average): {recall:.2f}%")

# Step 7: Plot Confusion Matrix
plt.figure(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["No Tumour", "Tumour"], yticklabels=["No Tumour", "Tumour"])
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")
plt.show()
joblib.dump(feature_extractor, "feature_extractor.pkl")
joblib.dump(scaler, "scaler.pkl")
joblib.dump(pca, "pca.pkl")
joblib.dump(log_reg_classifier, "logistic_regression_model.pkl")
feature_extractor = joblib.load("feature_extractor.pkl")
scaler = joblib.load("scaler.pkl")
pca = joblib.load("pca.pkl")
log_reg = joblib.load("logistic_regression_model.pkl")


#frontend ,backend code
import os
import numpy as np
import tensorflow as tf
import joblib
from flask import Flask, request, render_template
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# Initialize Flask App
app = Flask(__name__)

# Define the directory where trained models are stored
MODEL_DIR = os.path.join("static", "uploads")

# Define paths for the models
feature_extractor_path = os.path.join(MODEL_DIR, "feature_extractor.pkl")
scaler_path = os.path.join(MODEL_DIR, "scaler.pkl")
pca_path = os.path.join(MODEL_DIR, "pca.pkl")
log_reg_path = os.path.join(MODEL_DIR, "logistic_regression_model.pkl")

# Check if all model files exist
missing_files = [f for f in [feature_extractor_path, scaler_path, pca_path, log_reg_path] if not os.path.exists(f)]
if missing_files:
    raise FileNotFoundError(f" Missing model files in '{MODEL_DIR}': {missing_files}")

# Load the trained models
feature_extractor = joblib.load(feature_extractor_path)
scaler = joblib.load(scaler_path)
pca = joblib.load(pca_path)
log_reg = joblib.load(log_reg_path)

print(" Models loaded successfully!")

# Function to preprocess uploaded image
def preprocess_image(img_path):
    img = load_img(img_path, target_size=(224, 224))
    img_array = img_to_array(img) / 255.0  # Normalize
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    features = feature_extractor.predict(img_array)  # Extract features
    features_scaled = scaler.transform(features)  # Scale features
    features_pca = pca.transform(features_scaled)  # Apply PCA
    return features_pca

# Function to predict tumour
def predict_tumour(img_path):
    features_pca = preprocess_image(img_path)
    prediction = log_reg.predict(features_pca)[0]
    return "Tumour Detected" if prediction == 1 else "No Tumour Detected"

# Route for homepage
@app.route('/')
def home():
    return render_template("index.html")


# Route for uploading image
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return "No file uploaded", 400


    file = request.files['file']
    if file.filename == '':
        return "No selected file", 400

    # Save uploaded image
    file_path = os.path.join("static", "uploads", file.filename)
    file.save(file_path)

 # Get prediction
    result = predict_tumour(file_path)

    return render_template("result.html", image=file.filename, prediction=result)

if __name__ == "__main__":
    app.run(debug=True)


#file uploading code
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Brain Tumour Detection</title>
</head>
<body>
    <h1 style="text-align: center;">Brain Tumour Detection Centralised</h1>
    <form action="/predict" method="post" enctype="multipart/form-data">
        <input type="file" name="file" required>
        <button type="submit">Upload</button>
    </form>
</body>
</html>

#prediction result code
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Prediction Result</title>
</head>
<body>
    <h1>Brain Tumour Detection Result</h1>
    <img src="{{ url_for('static', filename='uploads/' + image) }}" alt="Uploaded Image" width="300">
    <h2>Prediction: {{ prediction }}</h2>
    <a href="/">Upload another image</a>
</body>
</html>






