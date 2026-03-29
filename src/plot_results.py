import matplotlib.pyplot as plt

# Fill your actual values here
models = [
    "Logistic Regression",
    "SVM",
    "CNN",
    "RNN",
    "BERT"
]

accuracies = [
    100,   # Logistic Regression (replace with actual)
    100,   # SVM (replace with actual)
    91.13,
    85.76,
    97.81
]

plt.figure()
plt.bar(models, accuracies)
plt.title("Model Accuracy Comparison")
plt.ylabel("Accuracy (%)")
plt.xticks(rotation=30)

plt.tight_layout()
plt.savefig("results/model_comparison.png")

print("Chart saved in results folder.")
plt.show()
