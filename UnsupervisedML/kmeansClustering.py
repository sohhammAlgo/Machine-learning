from sklearn.cluster import KMeans
import pandas as pd

df = pd.read_csv("../data/Pass-Fail.csv")

df['study_efficiency'] = df['study_hours_per_week']*df['attendance_pct']

X = df[['study_efficiency','homework_pct','midterm_score']]

model = KMeans(
    n_clusters=3,
    random_state=42
)

model.fit(X)

df['cluster'] = model.labels_

print(df.head())

new_student = [[8,90]]

cluster = model.predict(new_student)

print(cluster)