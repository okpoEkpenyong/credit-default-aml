"""
Load metrics/metrics.csv and plot comparison of accuracy & f1 for models.
Saves a PNG to outputs/metrics_comparison.png
"""
import pandas as pd
import matplotlib.pyplot as plt
import os

def plot_metrics(csv_path="metrics/metrics.csv", out_dir="outputs"):
    df = pd.read_csv(csv_path)
    df = df.groupby("model").mean().reset_index()
    os.makedirs(out_dir, exist_ok=True)
    fig, ax = plt.subplots(1,2, figsize=(10,4))
    ax[0].bar(df['model'], df['accuracy']); ax[0].set_title("Accuracy")
    ax[1].bar(df['model'], df['f1_score']); ax[1].set_title("F1 score")
    plt.tight_layout()
    out_path = os.path.join(out_dir, "metrics_comparison.png")
    fig.savefig(out_path)
    print("Saved metrics comparison to", out_path)

if __name__ == "__main__":
    plot_metrics()
