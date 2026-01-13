from sklearn.neighbors import KNeighborsClassifier


def get_model(X_train, y_train):
    model = KNeighborsClassifier()
    model.fit(X_train, y_train)

    return model
