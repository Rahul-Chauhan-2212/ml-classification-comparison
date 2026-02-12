from sklearn.ensemble import RandomForestClassifier

def get_model(random_state=42):
    return RandomForestClassifier(
        n_estimators=300,
        max_depth=12,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=random_state,
        n_jobs=-1
    )
