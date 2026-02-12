from sklearn.tree import DecisionTreeClassifier


def get_model(random_state=42):
    return DecisionTreeClassifier(
        max_depth=10,
        min_samples_split=10,
        min_samples_leaf=5,
        random_state=random_state
    )
