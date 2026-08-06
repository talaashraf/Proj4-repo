import matplotlib.pyplot as plt

def compare_models(nb_f1, lr_f1, emb_f1):

    models = [
        "Naive Bayes",
        "Logistic Regression",
        "Embeddings"
    ]

    f1_scores = [
        nb_f1,
        lr_f1,
        emb_f1
    ]

    plt.figure(figsize=(7,5))
    plt.bar(models, f1_scores)

    plt.title("Model Comparison")
    plt.xlabel("Models")
    plt.ylabel("F1-score")

    plt.savefig("model_comparison.png")
    plt.close()