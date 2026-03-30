from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier

def get_model(model_name):
    if model_name == "logreg":
        return LogisticRegression(
        solver="liblinear",
        max_iter=5000
    )
    
    elif model_name == "svm":
        return LinearSVC(max_iter=10000)

    elif model_name == "nb":
        return MultinomialNB()

    elif model_name == "rf":
        return RandomForestClassifier()

    else:
        raise ValueError("Invalid model name")


AVAILABLE_MODELS = ["logreg","svm","nb","rf"]
AVAILABLE_FEATURES = [ "bow","tfidf_uni","tfidf_uni_bi","char_ngrams"]