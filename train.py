import pickle
import numpy as np
from sklearn.datasets import fetch_lfw_people
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.calibration import CalibratedClassifierCV
import os

print("📂 Loading dataset...")
lfw = fetch_lfw_people(min_faces_per_person=20)

X = lfw.data
y_original = lfw.target
names = lfw.target_names

# ── Convert to Arnold vs Others ──────────────────────────
arnold_index = list(names).index("Arnold Schwarzenegger")
y = np.where(y_original == arnold_index, "Arnold", "Others")

# ── Balance the dataset ──────────────────────────────────
arnold_idx = np.where(y == "Arnold")[0]
others_idx = np.where(y == "Others")[0]

# Match Others count to Arnold count
np.random.seed(42)
others_sampled = np.random.choice(others_idx, size=len(arnold_idx), replace=False)
balanced_idx = np.concatenate([arnold_idx, others_sampled])

X_balanced = X[balanced_idx]
y_balanced = y[balanced_idx]

print(f"✅ Arnold images: {sum(y_balanced == 'Arnold')}")
print(f"✅ Others images: {sum(y_balanced == 'Others')}")

# ── PCA ──────────────────────────────────────────────────
print("\n🔬 Applying PCA (Eigenfaces)...")
pca = PCA(n_components=30, whiten=True, random_state=42)
X_pca = pca.fit_transform(X_balanced)

# ── Train/Test Split ─────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X_pca, y_balanced, test_size=0.2, random_state=42, stratify=y_balanced
)

# ── Train SVM (no deprecation warning) ───────────────────
print("🤖 Training SVM model...")
base_clf = SVC(kernel="rbf", C=10, gamma=0.001)
clf = CalibratedClassifierCV(base_clf, ensemble=False)
clf.fit(X_train, y_train)

# ── Evaluate ─────────────────────────────────────────────
y_pred = clf.predict(X_test)
print("\n📊 Results:")
print(classification_report(y_test, y_pred))

# ── Save Model ───────────────────────────────────────────
os.makedirs("model", exist_ok=True)
with open("model/model.pkl", "wb") as f:
    pickle.dump({"clf": clf, "pca": pca, "names": names}, f)

print("✅ Model saved to model/model.pkl")
