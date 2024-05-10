import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import precision_recall_curve, roc_auc_score, roc_curve
from sklearn.preprocessing import LabelBinarizer


def plot_confusion_matrix(confusion_matrix, labels):
    plt.figure(figsize=(4, 3))
    sns.heatmap(confusion_matrix, annot=True, fmt="d", cmap="Blues", cbar=False)
    plt.xlabel("Предсказания")
    plt.ylabel("Истинные значения")
    plt.title("Матрица ошибок")
    plt.xticks(ticks=np.arange(len(labels)) + 0.5, labels=labels, rotation=45)
    plt.yticks(ticks=np.arange(len(labels)) + 0.5, labels=labels, rotation=45)
    plt.show()
    plt.show()


def plot_roc_auc_curve(y_true, y_proba, labels):
    lb = LabelBinarizer()
    y_true_bin = lb.fit_transform(y_true)
    n_classes = y_true_bin.shape[1]
    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_proba[:, i])

        roc_auc[i] = roc_auc_score(y_true_bin[:, i], y_proba[:, i])

    # Визуализация ROC-AUC для каждого класса
    plt.figure(figsize=(4, 3))
    for i in range(n_classes):
        plt.plot(
            fpr[i],
            tpr[i],
            label="ROC curve (class {}) (area = {:.2f})".format(labels[i], roc_auc[i]),
        )
    plt.plot([0, 1], [0, 1], color="black", linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("График ROC-AUC")
    plt.legend(loc="lower right")
    plt.show()


def plot_prc_auc_curve(y_true, y_proba, labels):
    lb = LabelBinarizer()
    y_true_bin = lb.fit_transform(y_true)
    # Вычисление precision и recall для каждого класса
    n_classes = y_true_bin.shape[1]
    precision = dict()
    recall = dict()
    for i in range(n_classes):
        precision[i], recall[i], _ = precision_recall_curve(
            y_true_bin[:, i], y_proba[:, i]
        )

    # Визуализация Precision-Recall Curve для каждого класса
    plt.figure(figsize=(4, 3))
    for i in range(n_classes):
        plt.plot(recall[i], precision[i], label="PR curve (class {})".format(labels[i]))
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve for Multi-Class Classification")
    plt.legend(loc="lower left")
    plt.show()
