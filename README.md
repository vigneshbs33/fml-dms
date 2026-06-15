# DMS Functions Explorer: Live Digit Recognition (Group 9)

An interactive web application that bridges Discrete Mathematical Structures (DMS) and Machine Learning (ML) live. It trains a Multi-Layer Perceptron (MLP) Neural Network on the handwritten digits dataset and maps each phase of the pipeline directly to core function properties.

## Group 9 Details
1. Vignesh B S (Leader)
2. Rohit Maiya M
3. Sanskar Patil
4. Shailesh Nayak
5. Shreenivas CS

---

## 🚀 How to Run

1. Install the required Python packages:
   ```bash
   pip install numpy scikit-learn flask
   ```
2. Start the web application:
   ```bash
   python app.py
   ```
3. Open your browser and go to:
   ```
   http://127.0.0.1:5000
   ```

---

## 📐 DMS concepts Explained Live in the App

### 1. Cartesian Product ($F_1 \times F_2 \times \dots \times F_{64}$)
* **Where in App:** The 280x280 canvas.
* **Math Connection:** When you draw, Javascript downsamples the drawing to an $8 \times 8$ grid of values. Each value represents a dimension. The whole drawing is represented as a single 64-dimensional feature vector, which is an element of the Cartesian product space $\mathbb{R}^{64}$.

### 2. Plain Function ($f: X \to Y$)
* **Where in App:** The "DMS Mapping Result" box.
* **Math Connection:** A function maps every input to exactly one output. Here, the trained model takes your 64-dimensional feature vector and maps it deterministically to exactly one class label in $Y = \{0, 1, \dots, 9\}$.

### 3. Function Composition ($f_3 \circ f_2 \circ f_1$)
* **Where in App:** The middle "Network Vis" column.
* **Math Connection:** The network is built of layered transformations:
  - $f_1: \mathbb{R}^{64} \to \mathbb{R}^{32}$ (Hidden layer 1)
  - $f_2: \mathbb{R}^{32} \to \mathbb{R}^{16}$ (Hidden layer 2)
  - $f_3: \mathbb{R}^{16} \to \mathbb{R}^{10}$ (Output layer)
  - The model outputs $f(x) = (f_3 \circ f_2 \circ f_1)(x) = f_3(f_2(f_1(x)))$.

### 4. Pigeonhole Principle (Compression Limits)
* **Where in App:** The Pigeonhole card.
* **Math Connection:** The input space has $2^{64}$ (or $16^{64}$) possible shapes, whereas the output space has only 10 classes. Since the domain is vastly larger than the codomain, multiple distinct drawings *must* map to the same digit class.

### 5. Surjective (Onto) Check
* **Where in App:** The "Surjective Check" bar chart.
* **Math Connection:** A function is surjective if the Range equals the Codomain (meaning every output can be reached). The model uses Softmax at its output layer, which assigns non-zero probabilities to all 10 digits. Therefore, every digit remains reachable, preserving surjectivity.

### 6. Injective (1-to-1) Check
* **Where in App:** The "Injective Check" scatter plot.
* **Math Connection:** Injective means different inputs yield different outputs. Using PCA, we project the high-dimensional digits onto 2D space. The scatter plot shows distinct color clusters for each digit class, showing how they occupy separate regions in the internal feature space to preserve unique representations.

### 7. Inverse Function ($f^{-1}$)
* **Where in App:** The "Inverse Mapping" side-by-side view.
* **Math Connection:** A true mathematical inverse only exists for bijective functions. Since classification is not injective (Pigeonhole), we approximate the inverse $f^{-1}$ by retrieving the mean feature representation of the predicted class. The screen compares your raw drawing to the model's "ideal/average" representation of that digit.
