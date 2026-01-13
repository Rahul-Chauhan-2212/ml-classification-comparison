from sklearn.naive_bayes import GaussianNB


def get_model(X_train, y_train):
    model = GaussianNB()
    model.fit(X_train, y_train)

    return model
