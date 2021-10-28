# Baseline implementation - Lexical Overlap
# CISC867 Reproducibility study 2021
# Queen's University, Canada

# The following is an implementation of a bidirectional GRU network.
# This implementation is based on the PyTorch equivalent implemented by
# S. Panthaplackel, J.J. Li, G. Milos, and R.J. Mooney (2020),
# hosted at https://github.com/panthap2/deep-jit-inconsistency-detection

import json
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score


def get_data(file_location):
    with open(file_location) as data_file:
        data = json.load(data_file)
        data_file.close()
    return data


def evaluate(testy, result):
    accuracy = accuracy_score(testy, result)
    print("Accuracy: %f" % accuracy)
    # precision tp / (tp + fp)
    precision = precision_score(testy, result)
    print("Precision: %f" % precision)
    # recall: tp / (tp + fn)
    recall = recall_score(testy, result)
    print("Recall: %f" % recall)
    # f1: 2 tp / (2 tp + fp + fn)
    f1 = f1_score(testy, result)
    print("F1 score: %f" % f1)


class LexicalOverlap:
    def __init__(self):
        self.labels = []
        self.results = []
        self.data = get_data("data/Param/test.json")

    def generate_result(self):
        for value in self.data:
            # diff = value["token_diff_code_subtokens"]
            old_comment_st = value["old_comment_subtokens"]
            # diff = value["span_diff_code_subtokens"]
            diff = value["token_diff_code_subtokens"]
            label = value["label"]
            self.labels.append(label)
            inconsistent = 0

            # or "REPLACE_OLD" in diff
            if "<DELETE>" in diff:
                ind_delete = diff.index("<DELETE>")
                txt = diff[ind_delete + 1]
                if txt in old_comment_st:
                    inconsistent = 1

            elif "<REPLACE_OLD>" in diff:
                ind_rpl = diff.index("<REPLACE_OLD>")
                txt = diff[ind_rpl + 1]
                if txt in old_comment_st:
                    inconsistent = 1

            self.results.append(inconsistent)
        # for label in self.labels:
        #     if label == 1:
        #         count_label = count_label + 1
        #     else:
        #         count_label_zero = count_label_zero + 1
        evaluate(self.labels, self.results)
