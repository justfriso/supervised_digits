import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix

digits = load_digits()

X = digits.data / 16.0
y = digits.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#model = LogisticRegression()
model = KNeighborsClassifier(n_neighbors=5)
#model = SVC(kernel='linear')
#model = RandomForestClassifier(n_estimators=100)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

num_rows = 4
num_cols = 6
num_images = num_rows * num_cols

plt.figure(figsize=(12,8))
for i in range(num_images):
    plt.subplot(num_rows, num_cols, i + 1)
    plt.imshow(X_test[i].reshape(8, 8), cmap = 'gray')
    plt.title(f"True: {y_test[i]}, Pred: {y_pred[i]}")
    plt.axis("off")
    plt.tight_layout()
    plt.show
    
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
print("Classification Report:\n", classification_report(y_test, y_pred))
print("Confusion matrix:\n{}".format(confusion_matrix(y_test, y_pred)))

