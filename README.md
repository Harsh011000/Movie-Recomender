# 🎬 Movie Recommendation System

## 📌 Overview
This project is a **Movie Recommendation System** built using **TensorFlow, Flask, and Ray Tune** for hyperparameter tuning. It leverages a deep learning model with content-based filtering trained on user and movie feature data to predict movie ratings and provide recommendations.

## 🔥 Features
- **Preprocessing**: Scaling user and movie features using **MinMaxScaler**.
- **Model Training**: Deep learning model with **custom L2 normalization** layer.
- **Hyperparameter Optimization**: Utilizes **Ray Tune** for fine-tuning hyperparameters.
- **Flask API**: Exposes a `/predict` endpoint to get movie recommendations.
- **Deployment Ready**: Easily deployable as a web service.

## 📂 Project Structure
```
📦 Movie-Recommendation-System
├── 📜 Models/                      # Pretrained model and scalers
│   ├── my_model.keras              # Trained Keras model
│   ├── UserScaler.joblib           # Scaler for user features
│   ├── MovieScaler.joblib          # Scaler for movie features
│   ├── MinMaxRatingScaler.joblib   # Scaler for rating normalization
│
├── 📜 Data/                        # Dataset files
│   ├── updated_api_data.csv        # Movie feature dataset
│   ├── user_file_forApi.csv        # User feature dataset
│   ├── temp_user_data.csv          # Temporary user input data
│
├── 📜 Flask_Api.py                  # Flask application for predictions
├── 📜 requirements.txt               # Dependencies and packages required
├── 📜 README.md                      # Project documentation
└── 📜 .gitignore                      # Ignore unnecessary files
```

## 🚀 Getting Started
### 1️⃣ Install Dependencies
First, install the required Python libraries:
```sh
pip install -r requirements.txt
```

### 2️⃣ Run the Flask API
Start the Flask server using:
```sh
python Flask_Api.py
```
The API will be accessible at `http://127.0.0.1:5000/predict`.

### 3️⃣ Make a Prediction Request
Send a `GET` request with JSON input:
```json
{
  "data": [3.95, 4.25, 0.0, 0.0, 4.0, 4.12, 4.0, 4.04, 0.0, 3.0, 4.0, 0.0, 3.88, 3.89]
}
```
Example request using `cURL`:
```sh
curl -X GET "http://127.0.0.1:5000/predict" -H "Content-Type: application/json" -d '{"data": [3.95, 4.25, 0.0, 0.0, 4.0, 4.12, 4.0, 4.04, 0.0, 3.0, 4.0, 0.0, 3.88, 3.89]}'
```

### 4️⃣ Expected Response
The API returns a sorted list of movie recommendations:
```json
[
  {
    "prediction": 4.8,
    "label": "Highly Recommended",
    "title": "Inception",
    "genres": "Sci-Fi, Thriller",
    "year": 2010,
    "image_url": "https://example.com/inception.jpg"
  },
  ...
]
```

## 🔬 Model Architecture
- **Input Features**: User and movie feature embeddings.
- **Custom Layer**: `L2Normalization` for feature scaling.
- **Dense Layers**: Fully connected layers for rating prediction.
- **Activation**: ReLU activations with dropout for regularization.
- **Loss Function**: Mean Squared Error (MSE).

## 🛠️ Hyperparameter Tuning
Hyperparameter optimization is performed using **Ray Tune** with **Bayesian Optimization**. The tuning process explores:
- **Number of layers**: 2 to 5
- **Neurons per layer**: 32, 64, 128
- **Learning rate**: 0.0001 to 0.01
- **Batch size**: 32, 64, 128

### 📊 Example Ray Tune Configuration
```python
from ray import tune

def train_model(config):
    model = build_model(layers=config["layers"],
                        units=config["units"],
                        lr=config["learning_rate"])
    history = model.fit(train_data, train_labels,
                        batch_size=config["batch_size"],
                        epochs=10, validation_data=(val_data, val_labels))
    tune.report(loss=min(history.history['val_loss']))

search_space = {
    "layers": tune.choice([2, 3, 4, 5]),
    "units": tune.choice([32, 64, 128]),
    "learning_rate": tune.loguniform(0.0001, 0.01),
    "batch_size": tune.choice([32, 64, 128])
}

tuner = tune.run(train_model, config=search_space, num_samples=20)
```

## 📌 Dependencies
- Python 3.8+
- TensorFlow
- NumPy
- Pandas
- Flask
- Ray Tune
- Joblib

To install all dependencies, run:
```sh
pip install -r requirements.txt
```

