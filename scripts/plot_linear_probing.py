import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

plt.style.use("default")
sns.set_palette("husl")


DATA = {
    "experiment": [
        "ambient",
        "dance",
        "electronic",
        "hiphop",
        "jazz",
        "pop",
        "rock",
        "reggae",
        "classic_guitar",
        "bpm",
    ],
    "lr_act": [0.98, 0.97, 0.98, 1.0, 1.0, 0.98, 0.99, 1.0, 0.99, 0.8],
    "lr_sae32": [0.71, 0.72, 0.65, 0.67, 0.81, 0.67, 0.67, 0.57, 0.69, 0.58],
    "lr_sae512": [0.8, 0.81, 0.78, 0.82, 0.87, 0.75, 0.78, 0.89, 0.79, 0.66],
    "rf_act": [0.84, 0.83, 0.85, 0.87, 0.88, 0.8, 0.82, 0.89, 0.86, 0.77],
    "rf_sae32": [0.77, 0.8, 0.73, 0.75, 0.85, 0.73, 0.76, 0.55, 0.74, 0.72],
    "rf_sae512": [0.94, 0.94, 0.94, 0.95, 0.97, 0.93, 0.95, 0.94, 0.95, 0.96],
}


def main():
    df = pd.DataFrame(DATA)

    buckets = {
        "Activations": df["lr_act"].tolist() + df["rf_act"].tolist(),
        "SAE Features (top 32)": df["lr_sae32"].tolist() + df["rf_sae32"].tolist(),
        "SAE Features (top 512)": df["lr_sae512"].tolist() + df["rf_sae512"].tolist(),
    }

    df_buckets = pd.DataFrame([{"bucket": k, "value": v} for k, values in buckets.items() for v in values])

    plt.figure(figsize=(12, 8))

    sns.boxplot(x="bucket", y="value", data=df_buckets)

    plt.ylabel("Accurracy", fontsize=12)
    plt.xlabel("Data", fontsize=12)

    plt.title("Linear probing", fontsize=14, fontweight="bold")

    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    plt.savefig("linear_probing.png", dpi=300, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    main()
