
import pandas as pd

# Load evaluation dataset
df = pd.read_csv("../data/llm_evaluation_dataset.csv")

print("Total examples:", len(df))

print("\nPreference:")
print(df["Preference"].value_counts())

print("\nError types:")
print(df["Error_Type"].value_counts())

print("\nAverage scores:")
print(
    df[
        [
            "Accuracy_A",
            "Accuracy_B",
            "Relevance_A",
            "Relevance_B",
            "Instruction_A",
            "Instruction_B",
        ]
    ].mean()
)

print("\nHallucination cases:", (df["Hallucination"] == "Yes").sum())
print("Clear bias cases:", (df["Bias"] == "Clear Bias").sum())
