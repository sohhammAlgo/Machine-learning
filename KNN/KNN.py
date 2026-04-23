import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Sample dataset (you can replace with yours)
data = {
    "income":      [20,25,30,35,40,45,50,55,60,65,70,75],
    "credit_score":[300,400,500,550,600,650,700,720,750,780,800,820],
    "loan_amount": [5,10,15,20,25,30,35,40,45,50,55,60],
    "approved":    [0,0,0,0,1,1,1,1,1,1,1,1]
}

df = pd.DataFrame(data)

# Features & target
X = df[["income", "credit_score", "loan_amount"]]
y = df["approved"]

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model (k = 3)
model = KNeighborsClassifier(n_neighbors=3)
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Evaluation
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))