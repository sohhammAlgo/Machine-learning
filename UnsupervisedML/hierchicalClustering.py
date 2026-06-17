from sklearn.cluster import AgglomerativeClustering

import pandas as pd

df = pd.read_csv("../data/Pass-Fail.csv")

df['study_efficiency'] = df['study_hours_per_week']*df['attendance_pct']

X = df[['study_efficiency','homework_pct','midterm_score']]

model = AgglomerativeClustering(
    n_clusters=3
)

clusters = model.fit_predict(X)

df['cluster'] = clusters