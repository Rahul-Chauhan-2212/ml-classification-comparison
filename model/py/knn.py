from sklearn.neighbors import KNeighborsClassifier

def get_model():
    return KNeighborsClassifier(
        n_neighbors=7,
        weights="distance",
        metric="minkowski"
    )
