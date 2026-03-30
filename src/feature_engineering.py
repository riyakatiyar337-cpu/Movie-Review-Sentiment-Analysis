from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

def build_vectorizer(feature_type):

    if feature_type == "bow":
        return CountVectorizer(max_features=5000)

    elif feature_type == "tfidf_uni":
        return TfidfVectorizer(max_features=5000, ngram_range=(1,1))

    elif feature_type == "tfidf_uni_bi":
        return TfidfVectorizer(max_features=5000, ngram_range=(1,2))

    elif feature_type == "char_ngrams":
        return TfidfVectorizer(
            analyzer="char",
            ngram_range=(3,5),
            max_features=10000
        )

    else:
        raise ValueError("Invalid feature type")


def transform_text(vectorizer, X_train, X_test):
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)
    return X_train_vec, X_test_vec