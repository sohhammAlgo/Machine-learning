from sklearn.metrics import silhouette_score
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

score = silhouette_score(
    X,
    model.labels_
)

print(score)

for k in range(2,11):

    model = KMeans(
        n_clusters=k
    )

    model.fit(X)

    score = silhouette_score(
        X,
        model.labels_
    )

    print(k, score)