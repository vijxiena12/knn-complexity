from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, cross_val_score
import matplotlib.pyplot as plt

data = load_iris()
X = data.data
y = data.target 

#splitting the data to detect overfitting/underfitting
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

k_range = range(1, 31)
cv_scores = []
for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn, X_train, y_train, cv=5)
    print(f"k={k}, Cross-Validation Scores: {scores}, Mean Score: {scores.mean()}")
    cv_scores.append(scores.mean())

#visualize the 'sweet spot' for k
plt.figure(figsize=(8,5))
plt.plot(k_range, cv_scores, marker='o',color = 'green')
plt.xlabel('Value of k for KNN')
plt.ylabel('Cross-Validated Accuracy')
plt.title('Finding the Optimal k (Bias-Variance Tradeoff)')
plt.grid(True)
plt.show()