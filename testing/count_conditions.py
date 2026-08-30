"""Count the frequency of each disorder (Disease category) across the dataset."""

from pathlib import Path

import pandas as pd

DATASET_DIR = Path(__file__).resolve().parent.parent / "dataset"
CLASSES = ["malignant", "benign", "normal"]


def main():
    frames = []
    for cls in CLASSES:
        path = DATASET_DIR / cls / "medicalhistory.xlsx"
        df = pd.read_excel(path)
        df["class"] = cls
        frames.append(df)

    combined = pd.concat(frames, ignore_index=True)

    counts = combined["Disease category"].value_counts(dropna=False)
    print("Frequency of each condition (Disease category), all classes combined:\n")
    for condition, count in counts.items():
        print(f"{condition:<35} {count}")
    print(f"\nTotal records: {len(combined)}")

    print("\nPer-class breakdown:\n")
    breakdown = (
        combined.groupby(["class", "Disease category"]).size().rename("count")
    )
    print(breakdown.to_string())


if __name__ == "__main__":
    main()
