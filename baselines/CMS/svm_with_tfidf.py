# Baseline implementation - Lexical Overlap
# CISC867 Reproducibility study 2021
# Queen's University, Canada

# The following is an implementation of a bidirectional GRU network.
# This implementation is based on the PyTorch equivalent implemented by
# S. Panthaplackel, J.J. Li, G. Milos, and R.J. Mooney (2020),
# hosted at https://github.com/panthap2/deep-jit-inconsistency-detection

import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.svm import SVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score


# construct the column transfomer
def transfer_column(comment_column, code_column):
    column_transformer = ColumnTransformer(
        [('tfidf_comment', TfidfVectorizer(tokenizer=lambda i: i, lowercase=False), comment_column),
         ('tfidf_code', TfidfVectorizer(tokenizer=lambda i: i, lowercase=False), code_column)])

    return column_transformer


if __name__ == "__main__":
    # Set X and y
    # @todo: provide location of our dataset
    df = pd.read_json("data/Param/train.json")

    comment = "old_comment_subtokens"
    code = "old_code_subtokens"

    # comment = "old_comment_raw"
    # code = "old_code_raw"

    X = df[[comment,
            code]]
    # print(X)
    y = df["label"]

    model = SVC()

    column = transfer_column(comment, code)

    # print(column)

    pipe = Pipeline([
        ('tfidf', column),
        ('classify', model)
    ])

    # fit the model
    pipe.fit(X, y)

    # @todo: provide location of our dataset
    df_test = pd.read_json("data/Param/test.json")

    X_test = df_test[[comment,
                      code]]

    y_test = df_test["label"]

    y_pred = pipe.predict(X_test)

    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))
    print(accuracy_score(y_test, y_pred))
