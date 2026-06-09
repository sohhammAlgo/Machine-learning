import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

data = {
    "income":[20,25,30,35,40,45,50,55,60,65,70,75],
    "credit_score":[300,400,500,550,600,650,700,720,750,780,800,820],
    "loan_amount":[5,10,15,20,25,30,35,40,45,50,55,60],
    "approved":[0,0,0,0,1,1,1,1,1,1,1,1]
}

df = pd.DataFrame(data)

X = df[["income","credit_score","loan_amount"]]
y = df["approved"]

X_train,X_test,y_train,y_test = train_test_split(
    X,y,test_size=0.2,random_state=42
)

model = DecisionTreeClassifier(
    criterion='gini',
    max_depth=3
)

model.fit(X_train,y_train)

y_pred = model.predict(X_test)

print("Accuracy:",accuracy_score(y_test,y_pred))
print("Precision:",precision_score(y_test,y_pred))
print("Recall:",recall_score(y_test,y_pred))
print("F1:",f1_score(y_test,y_pred))

print("\nConfusion Matrix")
print(confusion_matrix(y_test,y_pred))

new_customer = [[50,700,35]]

prediction = model.predict(new_customer)

print(prediction[0])