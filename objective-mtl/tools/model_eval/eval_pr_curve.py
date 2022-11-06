import pickle
from sklearn.metrics import precision_recall_curve


def get_recall_with_fixed_precision(precisions, recalls, p_threshold):
    for i, prec in enumerate(precisions):
        if prec >= p_threshold:
            return recalls[i]
    return recalls[-1]


def get_precision_with_fixed_recall(precisions, recalls, p_recall):
    for i, rec in enumerate(recalls):
        if rec <= p_recall:
            return precisions[i]
    return precisions[-1]


if __name__ == "__main__":
    RES_PATH = "data/test_info/label_and_logits.pkl"
    y_true, y_score = pickle.load(RES_PATH)
    PTH = 0.8
    RTH = 0.9
    precs, recs, ths = precision_recall_curve(y_true, y_score)

    recall_res = get_recall_with_fixed_precision(precs, recs, PTH)
    precision_res = get_precision_with_fixed_recall(precs, recs, RTH)
