from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer


def create_bow_features(X_train_text, X_test_text):

    vectorizer = CountVectorizer()

    X_train = vectorizer.fit_transform(X_train_text)
    X_test = vectorizer.transform(X_test_text)

    return X_train, X_test, vectorizer


def create_tfidf_features(X_train_text, X_test_text):

    tfidf_vectorizer = TfidfVectorizer()

    X_train = tfidf_vectorizer.fit_transform(X_train_text)
    X_test = tfidf_vectorizer.transform(X_test_text)

    return X_train, X_test, tfidf_vectorizer