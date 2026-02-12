import pandas as pd
from sklearn.model_selection import train_test_split

# Load dataset
df = pd.read_csv("adult.csv")

# Split dataset (90% train, 10% test)
train_df, test_df = train_test_split(
    df,
    test_size=0.1,  # 10% test
    random_state=42,
    shuffle=True
)

# Save files
train_df.to_csv("train.csv", index=False)
test_df.to_csv("test.csv", index=False)

print("Dataset split complete!")
print(f"Train shape: {train_df.shape}")
print(f"Test shape: {test_df.shape}")
