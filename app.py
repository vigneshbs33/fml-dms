import os
import sys
sys.stdout.reconfigure(encoding='utf-8')
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import numpy as np
from sklearn.datasets import load_digits
from sklearn.neural_network import MLPClassifier
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)
CORS(app)

# Global variables
model = None
pca_2d = None
embedding_data = []
mean_images = {}
X_scaled_global = None
y_global = None

def init_model():
    global model, pca_2d, embedding_data, mean_images, X_scaled_global, y_global
    print("Initializing and training Digits MLP Model...")
    
    # Load scikit-learn digits dataset (8x8 images, 1797 samples)
    digits = load_digits()
    X, y = digits.data, digits.target
    
    # Scale feature values from [0, 16] to [0, 1] for better neural net training
    X_scaled = X / 16.0
    X_scaled_global = X_scaled
    y_global = y
    
    # Train the MLP Classifier
    # Composition: Input(64) -> Hidden(32) -> Hidden(16) -> Output(10)
    model = MLPClassifier(
        hidden_layer_sizes=(32, 16),
        activation='relu',
        max_iter=1000,
        random_state=42
    )
    model.fit(X_scaled, y)
    print(f"Model trained! Accuracy on training set: {model.score(X_scaled, y)*100:.2f}%")
    
    # Precompute PCA for 2D visual representation of Injective Embedding
    pca_2d = PCA(n_components=2, random_state=42)
    X_pca = pca_2d.fit_transform(X_scaled)
    
    # Sample 300 points for the embedding plot to keep it fast
    indices = np.random.choice(len(X_scaled), 300, replace=False)
    embedding_data = []
    for idx in indices:
        embedding_data.append({
            'x': float(X_pca[idx, 0]),
            'y': float(X_pca[idx, 1]),
            'label': int(y[idx])
        })
        
    # Precompute Inverse Mapping (mean image for each digit)
    for digit in range(10):
        digit_imgs = X_scaled[y == digit]
        mean_img = np.mean(digit_imgs, axis=0)
        mean_images[str(digit)] = mean_img.tolist()
        
    print("Precomputations complete.")

with app.app_context():
    init_model()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        # Expecting a list of 64 normalized float values in [0, 1]
        features = data['features']
        if len(features) != 64:
            return jsonify({'success': False, 'error': 'Invalid input length. Must be 64.'})
            
        features_arr = np.array(features).reshape(1, -1)
        
        # Predict
        prediction = int(model.predict(features_arr)[0])
        
        # Softmax outputs (probabilities)
        probs = model.predict_proba(features_arr)[0]
        
        # Compute projection of this input on the pre-trained PCA space
        pca_proj = pca_2d.transform(features_arr)[0]
        
        return jsonify({
            'success': True,
            'prediction': prediction,
            'probabilities': probs.tolist(),
            'pca_projection': pca_proj.tolist()
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/embedding-data', methods=['GET'])
def get_embedding_data():
    return jsonify({
        'success': True,
        'embeddings': embedding_data
    })

@app.route('/inverse-data', methods=['GET'])
def get_inverse_data():
    return jsonify({
        'success': True,
        'mean_images': mean_images
    })

if __name__ == '__main__':
    app.run(debug=True, port=5000)
