from sklearn.cluster import DBSCAN
import pandas as pd

df = pd.read_csv("../data/Pass-Fail.csv")

df['study_efficiency'] = df['study_hours_per_week']*df['attendance_pct']

X = df[['study_efficiency','homework_pct','midterm_score']]

model = DBSCAN(
    eps=0.5,
    min_samples=5
)

clusters = model.fit_predict(X)

print(clusters)