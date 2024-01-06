import pandas as pd

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.preprocessing import Normalizer


def vectorize_cleaned_text(vec_choice=1):
    """
    vectorizes cleaned text (from preprocess_text.py)

    :param vec_choice: vectorizer choice
            Choice 1: TFIDF (default)
            Chocie 2: Count
    :return: returns a list of lists
            Form: [features, indexes, max_iter, n-it]
            In addition, it returns true labels of training set, the list of all features,
            and the test data + labels
    """
    # CHANGE FILE NAME AS NEEDED
    df = pd.read_csv('instrument_clean.csv',
                     names=['', "text_raw", "instrument", "text_clean", "text_freq"], nrows=200)

    df_test = pd.read_csv('instrument_clean.csv',
                          names=['', "text_raw", "instrument", "text_clean", "text_freq"], skiprows=200)

    y_test = df_test["instrument"]
    y = df["instrument"]

    if vec_choice == 1:
        vectorizer = TfidfVectorizer()
    else:
        vectorizer = CountVectorizer()

    X = vectorizer.fit_transform(df['text_clean'].fillna(' '))
    X = Normalizer().fit_transform(X)
    X = X.todense()

    X_test = vectorizer.transform(df_test['text_clean'].fillna(' '))
    X_test = Normalizer().transform(X_test)

    return X, y, X_test, y_test