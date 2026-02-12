from sklearn.linear_model import LogisticRegression


def get_model():
    return LogisticRegression(
        max_iter=2000,
        C=1.5,
        solver="lbfgs"
    )
