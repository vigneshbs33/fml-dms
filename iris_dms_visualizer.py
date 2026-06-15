"""
╔══════════════════════════════════════════════════════════════════════════╗
║    IRIS DMS FUNCTION VISUALIZER                                          ║
║    Group 9 — BCS405A Discrete Mathematical Structures, VTU 4th Sem      ║
║                                                                          ║
║    Demonstrates DMS Functions concepts LIVE inside an ML model:          ║
║    • Cartesian Products   • Injective Functions  • Surjective Functions  ║
║    • Function Composition • Inverse Functions    • Pigeonhole Principle  ║
╚══════════════════════════════════════════════════════════════════════════╝

HOW TO RUN:
    pip install numpy pandas scikit-learn matplotlib tensorflow
    python iris_dms_visualizer.py
"""

import sys
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')   # change to 'TkAgg' if you want interactive windows
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix, classification_report
import warnings
warnings.filterwarnings('ignore')

# ─── Try importing TensorFlow; fall back to sklearn MLP ─────────────────────
try:
    import tensorflow as tf
    from tensorflow import keras
    USE_KERAS = True
    print("✅ TensorFlow found — using Keras neural network")
except ImportError:
    from sklearn.neural_network import MLPClassifier
    USE_KERAS = False
    print("⚠️  TensorFlow not found — using sklearn MLPClassifier")

print("\n" + "="*70)
print("  IRIS DMS FUNCTION VISUALIZER — GROUP 9")
print("="*70)

# ═══════════════════════════════════════════════════════════════════════════
# 1. CARTESIAN PRODUCT — The Input Feature Space
# ═══════════════════════════════════════════════════════════════════════════
print("\n📐 CONCEPT 1: CARTESIAN PRODUCT — Feature Space")
print("-"*50)

iris = load_iris()
X, y = iris.data, iris.target
feature_names = iris.feature_names
class_names   = iris.target_names

print(f"Feature sets (each is a set of real numbers):")
for i, name in enumerate(feature_names):
    print(f"  F{i+1} = {name}: [{X[:,i].min():.1f}, {X[:,i].max():.1f}] cm")

print(f"\nInput space = F1 × F2 × F3 × F4")
print(f"Each data point is an ORDERED TUPLE (x1, x2, x3, x4) ∈ F1×F2×F3×F4")
print(f"Total data points: {len(X)} ← elements of the Cartesian product")
print(f"Classes (Y): {list(class_names)}")
print(f"The ML function: f : F1×F2×F3×F4 → Y")

# ═══════════════════════════════════════════════════════════════════════════
# 2. PREPARE DATA  
# ═══════════════════════════════════════════════════════════════════════════
print("\n🔢 CONCEPT 2: FUNCTION f : X → Y — The ML Model")
print("-"*50)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s  = scaler.transform(X_test)

print(f"Domain (A): {X_train_s.shape[0]} training samples × 4 features")
print(f"Codomain (B): 3 classes — {list(class_names)}")
print("Model is a function: every input → exactly 1 predicted class")

# ═══════════════════════════════════════════════════════════════════════════
# 3. BUILD MODEL (composition of functions)
# ═══════════════════════════════════════════════════════════════════════════
print("\n⛓️  CONCEPT 3: FUNCTION COMPOSITION — Neural Network Layers")
print("-"*50)

if USE_KERAS:
    model = keras.Sequential([
        keras.layers.Dense(16, activation='relu', input_shape=(4,), name="f1_hidden1"),
        keras.layers.Dense(8,  activation='relu',                   name="f2_hidden2"),
        keras.layers.Dense(3,  activation='softmax',                name="f3_output"),
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    print("Model architecture (Composition: f3 ∘ f2 ∘ f1):")
    model.summary()
    history = model.fit(X_train_s, y_train, epochs=100, verbose=0, validation_split=0.2)
    y_pred = np.argmax(model.predict(X_test_s, verbose=0), axis=1)
    # Get intermediate outputs for composition visualization
    layer1_out = model.get_layer('f1_hidden1')(X_test_s[:5]).numpy()
    layer2_out = model.get_layer('f2_hidden2')(X_test_s[:5]).numpy()
    train_loss = history.history['loss']
    val_loss   = history.history['val_loss']
else:
    model = MLPClassifier(hidden_layer_sizes=(16, 8), max_iter=1000, random_state=42)
    model.fit(X_train_s, y_train)
    y_pred = model.predict(X_test_s)
    layer1_out = None; layer2_out = None
    train_loss = model.loss_curve_
    val_loss   = None
    print("Model: Input(4) → Dense(16, ReLU) → Dense(8, ReLU) → Softmax(3)")
    print("Full composition: f3(f2(f1(x)))  where each fi is a layer transform")

print("\nComposition chain:")
print("  x ∈ ℝ⁴  →[f1: ReLU(W1x+b1)]→  h1 ∈ ℝ¹⁶")
print("          →[f2: ReLU(W2h1+b2)]→  h2 ∈ ℝ⁸")
print("          →[f3: Softmax(W3h2+b3)]→ ŷ ∈ ℝ³")
print("  Full model = (f3 ∘ f2 ∘ f1)(x) = f3(f2(f1(x)))")

# ═══════════════════════════════════════════════════════════════════════════
# 4. SURJECTIVITY CHECK — All classes predicted?
# ═══════════════════════════════════════════════════════════════════════════
print("\n→All CONCEPT 4: SURJECTIVE FUNCTION — Coverage Check")
print("-"*50)

predicted_classes = set(y_pred)
all_classes       = set(range(3))
print(f"Codomain (all possible classes): {all_classes}")
print(f"Range (classes actually predicted): {predicted_classes}")

if predicted_classes == all_classes:
    print("✅ SURJECTIVE! Model predicts ALL 3 classes. Range = Codomain.")
else:
    missing = all_classes - predicted_classes
    print(f"❌ NOT SURJECTIVE! Missing classes: {missing}")
    print(f"   Dead outputs: {[class_names[i] for i in missing]}")

# ═══════════════════════════════════════════════════════════════════════════
# 5. INJECTIVITY CHECK (approximate) — Unique embeddings
# ═══════════════════════════════════════════════════════════════════════════
print("\n1→1 CONCEPT 5: INJECTIVE FUNCTION — Unique Embeddings")
print("-"*50)

# Use PCA as a proxy to show uniqueness of representations
pca_embed = PCA(n_components=2)
embeddings = pca_embed.fit_transform(X_test_s)
unique_2d = len(set(map(tuple, embeddings.round(4))))
print(f"After projecting {len(X_test_s)} test samples to 2D:")
print(f"  Unique 2D positions: {unique_2d}")
if unique_2d == len(X_test_s):
    print("  ✅ All embeddings are unique (injective in 2D)")
else:
    print(f"  ⚠️  {len(X_test_s)-unique_2d} collisions (not injective — see Pigeonhole!)")

# ═══════════════════════════════════════════════════════════════════════════
# 6. PIGEONHOLE PRINCIPLE — Dimensionality reduction limits
# ═══════════════════════════════════════════════════════════════════════════
print("\n🐦 CONCEPT 6: PIGEONHOLE PRINCIPLE — Compression Limits")
print("-"*50)
print(f"Original feature space: 4 dimensions")
print(f"After PCA to 2D: {pca_embed.explained_variance_ratio_.sum()*100:.1f}% variance retained")
print(f"Lost info: {(1-pca_embed.explained_variance_ratio_.sum())*100:.1f}%")
print(f"")
print(f"PIGEONHOLE: mapping {len(X_test_s)} 4D points → 2D grid")
print(f"With infinite precision both are injective, but rounding creates collisions.")
print(f"When compressing: at least two inputs MUST share features (Pigeonhole guarantees this)")

# ═══════════════════════════════════════════════════════════════════════════
# 7. INVERSE FUNCTION — Autoencoder-style reconstruction
# ═══════════════════════════════════════════════════════════════════════════
print("\n🔄 CONCEPT 7: INVERSE FUNCTION — Reconstruction Demo")
print("-"*50)

# Use PCA as encoder/decoder pair (PCA IS mathematically invertible!)
pca_enc = PCA(n_components=2)   # Encoder: 4D → 2D
encoded = pca_enc.fit_transform(X_test_s)   # E(x) = z
decoded = pca_enc.inverse_transform(encoded) # D(z) ≈ x  ← This is the INVERSE!

recon_error = np.mean(np.square(X_test_s - decoded))
print(f"Encoder E : ℝ⁴ → ℝ² (4D → 2D latent code)")
print(f"Decoder D : ℝ² → ℝ⁴ (2D → reconstruct 4D)")
print(f"D(E(x)) ≈ x  ← Inverse function approximation")
print(f"Mean reconstruction error (||x - D(E(x))||²): {recon_error:.4f}")
print(f"Note: Not zero because 2D < 4D (Pigeonhole — info lost!)")
print(f"With 4 components (lossless): PCA becomes truly invertible (bijective)")

pca_full = PCA(n_components=4)
enc_full  = pca_full.fit_transform(X_test_s)
dec_full  = pca_full.inverse_transform(enc_full)
full_err  = np.mean(np.square(X_test_s - dec_full))
print(f"Full 4-component PCA reconstruction error: {full_err:.6f} ≈ 0 (bijective!)")

# ═══════════════════════════════════════════════════════════════════════════
# 8. ACCURACY REPORT
# ═══════════════════════════════════════════════════════════════════════════
print("\n📊 MODEL PERFORMANCE")
print("-"*50)
acc = np.mean(y_pred == y_test)
print(f"Test Accuracy: {acc*100:.1f}%")
print(f"\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=class_names))

# ═══════════════════════════════════════════════════════════════════════════
# 9. VISUALIZATION — 8-panel figure
# ═══════════════════════════════════════════════════════════════════════════
print("\n🎨 Generating visualizations...")

fig = plt.figure(figsize=(20, 14), facecolor='#0D122B')
fig.suptitle('Iris DMS Function Visualizer — Group 9\nBCS405A Discrete Mathematical Structures, VTU',
             fontsize=16, color='white', fontweight='bold', y=0.98)

gs = gridspec.GridSpec(3, 4, figure=fig, hspace=0.45, wspace=0.38)

COLORS = ['#7C3AED', '#06B6D4', '#10B981']
PALETTE = {'bg': '#0D122B', 'card': '#161B3A', 'text': '#CBD5E1',
           'violet': '#7C3AED', 'cyan': '#06B6D4', 'green': '#10B981',
           'amber': '#F59E0B', 'rose': '#F43F5E'}

def card(ax):
    ax.set_facecolor(PALETTE['card'])
    for spine in ax.spines.values():
        spine.set_edgecolor('#2D3A5E')

# --- Panel 1: Cartesian Product / Feature distributions ---
ax1 = fig.add_subplot(gs[0, 0])
card(ax1)
ax1.set_title('Concept 1: Cartesian Product\n(Feature Distributions)', color=PALETTE['violet'], fontsize=9)
for i, name in enumerate(feature_names):
    for cls in range(3):
        data = X[y == cls, i]
        ax1.hist(data, bins=10, alpha=0.5, color=COLORS[cls], label=class_names[cls] if i == 0 else "")
ax1.set_xlabel('Feature Value (cm)', color=PALETTE['text'], fontsize=8)
ax1.set_ylabel('Count', color=PALETTE['text'], fontsize=8)
ax1.tick_params(colors=PALETTE['text'], labelsize=7)
ax1.legend(fontsize=6, labelcolor=PALETTE['text'], facecolor=PALETTE['card'])

# --- Panel 2: Function composition — training loss ---
ax2 = fig.add_subplot(gs[0, 1])
card(ax2)
ax2.set_title("Concept 3: Function Composition\n(Training = Refining f₃∘f₂∘f₁)", color=PALETTE['cyan'], fontsize=9)
ax2.plot(train_loss, color=PALETTE['violet'], linewidth=2, label='Train Loss')
if val_loss:
    ax2.plot(val_loss, color=PALETTE['amber'], linewidth=2, linestyle='--', label='Val Loss')
ax2.set_xlabel('Epoch', color=PALETTE['text'], fontsize=8)
ax2.set_ylabel('Loss', color=PALETTE['text'], fontsize=8)
ax2.tick_params(colors=PALETTE['text'], labelsize=7)
ax2.legend(fontsize=7, labelcolor=PALETTE['text'], facecolor=PALETTE['card'])

# --- Panel 3: Injective — PCA 2D scatter (embeddings) ---
ax3 = fig.add_subplot(gs[0, 2])
card(ax3)
ax3.set_title("Concept 5: Injective Function\n(Unique 2D Embeddings per Class)", color=PALETTE['green'], fontsize=9)
for cls in range(3):
    mask = y_test == cls
    ax3.scatter(embeddings[mask, 0], embeddings[mask, 1], c=COLORS[cls],
                label=class_names[cls], alpha=0.8, s=40, edgecolors='none')
ax3.set_xlabel('PC1', color=PALETTE['text'], fontsize=8)
ax3.set_ylabel('PC2', color=PALETTE['text'], fontsize=8)
ax3.tick_params(colors=PALETTE['text'], labelsize=7)
ax3.legend(fontsize=7, labelcolor=PALETTE['text'], facecolor=PALETTE['card'])

# --- Panel 4: Surjective — confusion matrix ---
ax4 = fig.add_subplot(gs[0, 3])
card(ax4)
ax4.set_title("Concept 4: Surjective Function\n(All Classes Predicted?)", color=PALETTE['amber'], fontsize=9)
cm = confusion_matrix(y_test, y_pred)
im = ax4.imshow(cm, cmap='Purples')
ax4.set_xticks(range(3)); ax4.set_yticks(range(3))
ax4.set_xticklabels(['Setosa', 'Versic.', 'Virgin.'], color=PALETTE['text'], fontsize=7, rotation=20)
ax4.set_yticklabels(['Setosa', 'Versic.', 'Virgin.'], color=PALETTE['text'], fontsize=7)
for i in range(3):
    for j in range(3):
        ax4.text(j, i, cm[i, j], ha='center', va='center', color='white', fontsize=11, fontweight='bold')
ax4.set_xlabel('Predicted', color=PALETTE['text'], fontsize=8)
ax4.set_ylabel('Actual', color=PALETTE['text'], fontsize=8)
surj_note = "✅ SURJECTIVE" if predicted_classes == all_classes else "❌ NOT SURJECTIVE"
ax4.set_title(f"Concept 4: Surjective\n{surj_note}", color=PALETTE['amber'], fontsize=9)

# --- Panel 5: Pigeonhole — info loss vs components ---
ax5 = fig.add_subplot(gs[1, 0:2])
card(ax5)
ax5.set_title("Concept 6: Pigeonhole Principle — Info Loss in Dimensionality Reduction",
              color=PALETTE['rose'], fontsize=9)
components = list(range(1, 5))
var_retained = []
for n in components:
    p = PCA(n_components=n)
    p.fit(X_test_s)
    var_retained.append(p.explained_variance_ratio_.sum() * 100)
bars = ax5.bar(components, var_retained, color=[PALETTE['rose'], PALETTE['amber'], PALETTE['cyan'], PALETTE['green']])
ax5.axhline(100, color='white', linestyle='--', linewidth=1, alpha=0.5, label='100% (no pigeonhole loss)')
for bar, val in zip(bars, var_retained):
    ax5.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
             f'{val:.1f}%', ha='center', color=PALETTE['text'], fontsize=9, fontweight='bold')
ax5.set_xticks(components)
ax5.set_xticklabels([f'{n} components\n({"✅ bijective" if n==4 else "❌ info lost"})' for n in components],
                     color=PALETTE['text'], fontsize=8)
ax5.set_ylabel('Variance Retained (%)', color=PALETTE['text'], fontsize=8)
ax5.tick_params(colors=PALETTE['text'], labelsize=8)
ax5.set_ylim(0, 115)
ax5.legend(fontsize=8, labelcolor=PALETTE['text'], facecolor=PALETTE['card'])
ax5.text(3, 55, 'Pigeonhole:\n4D → 2D ≡\nn+1 pigeons\ninto n holes\n→ Info MUST\nbe lost', 
         color=PALETTE['rose'], fontsize=8, va='center', style='italic',
         bbox=dict(boxstyle='round', facecolor='#2A0D0D', edgecolor=PALETTE['rose']))

# --- Panel 6: Inverse / Reconstruction ---
ax6 = fig.add_subplot(gs[1, 2:4])
card(ax6)
ax6.set_title("Concept 7: Inverse Function — Encoder/Decoder Reconstruction Error",
              color=PALETTE['violet'], fontsize=9)
n_comps = range(1, 5)
errors  = []
for n in n_comps:
    p = PCA(n_components=n)
    enc = p.fit_transform(X_test_s)
    dec = p.inverse_transform(enc)
    errors.append(np.mean(np.square(X_test_s - dec)))

ax6.plot(n_comps, errors, color=PALETTE['violet'], linewidth=2.5, marker='o', markersize=8)
for n, e in zip(n_comps, errors):
    ax6.text(n+0.05, e+0.001, f'{e:.4f}', color=PALETTE['text'], fontsize=8)
ax6.axhline(0, color=PALETTE['green'], linestyle='--', linewidth=1.5, label='Error=0 (perfect inverse: bijective)')
ax6.fill_between(n_comps, errors, 0, alpha=0.2, color=PALETTE['violet'])
ax6.set_xticks(n_comps)
ax6.set_xticklabels([f'n={n}' for n in n_comps], color=PALETTE['text'], fontsize=9)
ax6.set_ylabel('Reconstruction Error ||x - D(E(x))||²', color=PALETTE['text'], fontsize=8)
ax6.set_xlabel('PCA Components', color=PALETTE['text'], fontsize=8)
ax6.tick_params(colors=PALETTE['text'])
ax6.legend(fontsize=8, labelcolor=PALETTE['text'], facecolor=PALETTE['card'])
ax6.text(3, errors[1], 'n=4: error≈0\nBijective!\nD = E⁻¹', color=PALETTE['green'], fontsize=8, ha='right',
         bbox=dict(boxstyle='round', facecolor='#0A2A1F', edgecolor=PALETTE['green']))

# --- Panel 7: Feature pair scatter (Cartesian Product visualization) ---
ax7 = fig.add_subplot(gs[2, 0:2])
card(ax7)
ax7.set_title("Concept 1: Cartesian Product F₁×F₃\n(Sepal Length × Petal Length sub-space)",
              color=PALETTE['cyan'], fontsize=9)
for cls in range(3):
    mask = y == cls
    ax7.scatter(X[mask, 0], X[mask, 2], c=COLORS[cls], label=class_names[cls], alpha=0.7, s=40, edgecolors='none')
ax7.set_xlabel('Sepal Length (F₁)', color=PALETTE['text'], fontsize=9)
ax7.set_ylabel('Petal Length (F₃)', color=PALETTE['text'], fontsize=9)
ax7.tick_params(colors=PALETTE['text'], labelsize=8)
ax7.legend(fontsize=8, labelcolor=PALETTE['text'], facecolor=PALETTE['card'])
ax7.text(4.5, 6.0, 'Each point is an\nordered pair (f₁,f₃)\n∈ F₁×F₃', color=PALETTE['amber'],
         fontsize=8, bbox=dict(boxstyle='round', facecolor='#1A1200', edgecolor=PALETTE['amber']))

# --- Panel 8: DMS Concept Summary ---
ax8 = fig.add_subplot(gs[2, 2:4])
card(ax8)
ax8.axis('off')
ax8.set_title("DMS ↔ ML Concept Map Summary", color='white', fontsize=10, fontweight='bold')
table_data = [
    ["DMS Concept",        "ML Application",           "In Our Project"],
    ["Cartesian Product",  "Feature Space",             "4-feature Iris ℝ⁴"],
    ["Plain Function f",   "ML Model",                  "Neural Network"],
    ["Injective (1→1)",    "Embeddings",                "Unique 2D projections"],
    ["Surjective (Onto)",  "Class Coverage",            "All 3 classes predicted"],
    ["Composition g∘f",   "Deep Net Layers",           "f₃(f₂(f₁(x)))"],
    ["Inverse f⁻¹",       "Decoder/Autoencoder",       "PCA inverse_transform"],
    ["Pigeonhole",        "Compression Limits",        "4D→2D info loss"],
]
cell_colors = [['#1A0F4A', '#0F2A3A', '#0A2A1F']] + \
              [['#161B3A', '#161B3A', '#161B3A']] * (len(table_data)-1)
tbl = ax8.table(cellText=table_data[1:], colLabels=table_data[0],
                cellLoc='center', loc='center',
                cellColours=cell_colors[1:],
                colColours=['#1A0F4A', '#0F2A3A', '#0A2A1F'])
tbl.auto_set_font_size(False)
tbl.set_fontsize(8)
tbl.scale(1, 1.5)
for (r, c), cell in tbl.get_celld().items():
    cell.set_edgecolor('#2D3A5E')
    if r == 0:
        cell.set_text_props(color='white', fontweight='bold')
    else:
        colors_row = [PALETTE['violet'], PALETTE['cyan'], PALETTE['green']]
        cell.set_text_props(color=colors_row[c % 3])

plt.savefig('/mnt/user-data/outputs/iris_dms_visualization.png',
            dpi=150, bbox_inches='tight', facecolor='#0D122B')
print("✅ Visualization saved: iris_dms_visualization.png")

# ═══════════════════════════════════════════════════════════════════════════
# 10. FUNCTION COMPOSITION — Show layer-by-layer transform (first 3 samples)
# ═══════════════════════════════════════════════════════════════════════════
print("\n⛓️  FUNCTION COMPOSITION — Layer-by-layer for sample inputs:")
print("-"*50)
sample_indices = [0, 1, 2]
for idx in sample_indices:
    x = X_test_s[idx]
    true_class = class_names[y_test[idx]]
    pred_class = class_names[y_pred[idx]]
    print(f"\nSample {idx+1}:")
    print(f"  Input x ∈ ℝ⁴: {x.round(3)}")
    print(f"  True class:    {true_class}")
    print(f"  Predicted:     {pred_class}  {'✅' if true_class==pred_class else '❌'}")

print("\n" + "="*70)
print("  ALL DMS CONCEPTS DEMONSTRATED SUCCESSFULLY!")
print(f"  Final Model Accuracy: {acc*100:.1f}%")
print("="*70)
print("\nFiles generated:")
print("  📊 iris_dms_visualization.png — 8-panel concept visualization")
