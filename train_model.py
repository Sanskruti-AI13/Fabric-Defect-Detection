import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import pickle

images = []
labels = []

# Defect Free images
path = "dataset/archive (3)/images/defect_free"

for file in __import__("os").listdir(path):
    image = cv2.imread(path + "/" + file)
    image = cv2.resize(image, (64, 64))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = cv2.medianBlur(image, 3)
    image = cv2.equalizeHist(image)

    images.append(image.flatten())
    labels.append(0)

# Stain images
path = "dataset/archive (3)/images/stain"

for file in __import__("os").listdir(path):
    image = cv2.imread(path + "/" + file)
    image = cv2.resize(image, (64, 64))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = cv2.medianBlur(image, 3)
    image = cv2.equalizeHist(image)

    images.append(image.flatten())
    labels.append(1)

X = np.array(images)
y = np.array(labels)

print("Total images:", len(X))
print("Defect Free:", labels.count(0))
print("Stain:", labels.count(1))

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

model = DecisionTreeClassifier(
    class_weight="balanced",
    random_state=42
)

model.fit(X_train, y_train)

predictions = model.predict(X_test)

print("\nAccuracy:", accuracy_score(y_test, predictions))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, predictions))

with open("models/decision_tree.pkl", "wb") as file:
    pickle.dump(model, file)

print("\nModel saved successfully!")