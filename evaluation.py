from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)
import matplotlib.pyplot as plt
import seaborn as sns


def evaluate_model(model_name, y_test, predictions):
    accuracy = accuracy_score(y_test, predictions)

    precision = precision_score(
        y_test,
        predictions,
        average="weighted"
    )

    recall = recall_score(
        y_test,
        predictions,
        average="weighted"
    )

    f1 = f1_score(
        y_test,
        predictions,
        average="weighted"
    )

    print(f"\n========== {model_name} ==========")

    print("\nPredicted Labels:")
    print(predictions[:10])

    print("\nAccuracy:")
    print(accuracy)

    print("\nPrecision:")
    print(precision)

    print("\nRecall:")
    print(recall)

    print("\nF1-score:")
    print(f1)

    print("\nClassification Report:")
    print(classification_report(y_test, predictions))

    print("\nConfusion Matrix:")
    cm = confusion_matrix(y_test, predictions)
    print(cm)
    sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues"
       )

    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.show()
    


    return accuracy, precision, recall, f1