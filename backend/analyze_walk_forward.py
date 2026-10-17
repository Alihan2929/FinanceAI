
from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "walk_forward_results.csv"

df = pd.read_csv(DATA_PATH)

metrics = [
    "Accuracy",
    "Balanced Accuracy",
    "Precision",
    "Recall",
    "F1",
]

print("\n=== DÖNEM BAZINDA MODEL KARŞILAŞTIRMASI ===")

for metric in metrics:
    print(f"\n--- {metric} ---")

    table = df.pivot(
        index="Fold",
        columns="Model",
        values=metric,
    )

    print(table.round(3).to_string())

print("\n=== RANDOM FOREST EN İYİ / EN KÖTÜ DÖNEMLER ===")

rf = df[df["Model"] == "Random Forest"].copy()

print("\nEn iyi 3 dönem:")
print(
    rf.nlargest(3, "Balanced Accuracy")[
        ["Fold", "Test_Start", "Test_End", "Balanced Accuracy", "F1"]
    ].to_string(index=False)
)

print("\nEn kötü 3 dönem:")
print(
    rf.nsmallest(3, "Balanced Accuracy")[
        ["Fold", "Test_Start", "Test_End", "Balanced Accuracy", "F1"]
    ].to_string(index=False)
)
