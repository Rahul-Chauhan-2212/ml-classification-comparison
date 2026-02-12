from sklearn.naive_bayes import GaussianNB

def get_model():
    return GaussianNB(
        var_smoothing=1e-8
    )
