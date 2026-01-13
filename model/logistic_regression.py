from sklearn.linear_model import LogisticRegression


def get_model(X_train, y_train):
    model = LogisticRegression(max_iter=10000)
    model.fit(X_train, y_train)

    return model
