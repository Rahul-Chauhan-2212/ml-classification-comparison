from ucimlrepo import fetch_ucirepo
import pandas as pd

data = fetch_ucirepo(id=17)
X = data.data.features
y = data.data.targets

df = pd.concat([y, X], axis=1)
print("Dataset Shape is:", df.shape)
df.to_csv("breast_cancer.csv", index=False)
