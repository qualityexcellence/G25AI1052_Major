from sklearn.datasets import fetch_olivetti_faces
from sklearn.model_selection import train_test_split
import joblib
from sklearn.metrics import accuracy_score

data = fetch_olivetti_faces()

X = data.data
y = data.target

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.3,
    random_state=42
)

model = joblib.load("savedmodel.pth")

pred = model.predict(X_test)

acc = accuracy_score(y_test, pred)

print(f"Test Accuracy: {acc}")