
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Project paths
ROOT = Path(__file__).resolve().parent.parent
RAW_FILE = ROOT / "data" / "raw" / "Dataset-SA.csv"
PROCESSED_DIR = ROOT / "data" / "processed"
CHART_DIR = ROOT / "visualizations"

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
CHART_DIR.mkdir(parents=True, exist_ok=True)

# 1. Load data
df = pd.read_csv(RAW_FILE)

print("=" * 60)
print("INITIAL DATASET")
print("=" * 60)
print("Shape:", df.shape)
print("\nData types:\n", df.dtypes)
print("\nMissing values:\n", df.isna().sum())
print("\nExact duplicate rows:", df.duplicated().sum())

# 2. Standardize column names
df.columns = df.columns.str.strip().str.lower()

# 3. Convert numeric columns
df["product_price"] = pd.to_numeric(
    df["product_price"].astype(str).str.replace(",", "", regex=False),
    errors="coerce"
)

df["rate"] = pd.to_numeric(df["rate"], errors="coerce")

# 4. Standardize text columns
text_columns = ["product_name", "review", "summary", "sentiment"]

for col in text_columns:
    df[col] = df[col].astype("string").str.strip()

df["sentiment"] = df["sentiment"].str.lower()

# Treat blank strings as missing
for col in text_columns:
    df[col] = df[col].replace("", pd.NA)

# 5. Remove exact duplicate records
before = len(df)
df = df.drop_duplicates().copy()
print("\nExact duplicate rows removed:", before - len(df))

# 6. Flag invalid values before filtering
invalid_price = df["product_price"].isna() | (df["product_price"] <= 0)
invalid_rating = ~df["rate"].isin([1, 2, 3, 4, 5])
missing_product = df["product_name"].isna()

print("\nInvalid/missing prices:", int(invalid_price.sum()))
print("Invalid/missing ratings:", int(invalid_rating.sum()))
print("Missing product names:", int(missing_product.sum()))

# Ratings and prices must be valid for the core analysis.
# Missing review text is retained; it is not a missing rating.
df = df.loc[
    ~invalid_price & ~invalid_rating & ~missing_product
].copy()

df["rate"] = df["rate"].astype(int)

# 7. Engineer features
df["review_length"] = df["review"].fillna("").str.len()
df["summary_length"] = df["summary"].fillna("").str.len()

df["rating_group"] = pd.cut(
    df["rate"],
    bins=[0, 2, 3, 5],
    labels=["Low (1-2)", "Medium (3)", "High (4-5)"]
)

# Identify extreme review lengths using the IQR rule.
q1 = df["review_length"].quantile(0.25)
q3 = df["review_length"].quantile(0.75)
iqr = q3 - q1
upper_limit = q3 + 1.5 * iqr

df["review_length_outlier"] = (
    (df["review_length"] > upper_limit)
    if iqr > 0
    else False
)

print("\nCleaned dataset shape:", df.shape)
print("Missing review texts retained:", int(df["review"].isna().sum()))
print("Extreme review-length records flagged:",
      int(df["review_length_outlier"].sum()))

# 8. Save cleaned data
output_file = PROCESSED_DIR / "product_reviews_cleaned.csv"
df.to_csv(output_file, index=False)
print("\nSaved:", output_file)

# 9. EDA: rating distribution
sns.set_theme(style="whitegrid")

plt.figure(figsize=(8, 5))
sns.countplot(data=df, x="rate")
plt.title("Distribution of Product Ratings")
plt.xlabel("Rating")
plt.ylabel("Number of Reviews")
plt.tight_layout()
plt.savefig(CHART_DIR / "rating_distribution.png", dpi=150)
plt.close()

# 10. EDA: sentiment distribution
plt.figure(figsize=(8, 5))
sns.countplot(
    data=df,
    x="sentiment",
    order=df["sentiment"].value_counts().index
)
plt.title("Distribution of Review Sentiment")
plt.xlabel("Sentiment")
plt.ylabel("Number of Reviews")
plt.xticks(rotation=20)
plt.tight_layout()
plt.savefig(CHART_DIR / "sentiment_distribution.png", dpi=150)
plt.close()

# 11. EDA: average rating by sentiment
sentiment_summary = (
    df.groupby("sentiment", dropna=False)["rate"]
    .agg(["count", "mean", "median"])
    .round(2)
)
print("\nRating statistics by sentiment:\n", sentiment_summary)

# 12. EDA: product price distribution
plt.figure(figsize=(8, 5))
sns.histplot(
    data=df,
    x="product_price",
    bins=40
)
plt.title("Product Price Distribution")
plt.xlabel("Product Price")
plt.ylabel("Number of Reviews")
plt.tight_layout()
plt.savefig(CHART_DIR / "product_price_distribution.png", dpi=150)
plt.close()

# 13. Save a summary of actual results
with open(PROCESSED_DIR / "eda_summary.txt", "w", encoding="utf-8") as f:
    f.write(f"Cleaned rows: {len(df)}\n")
    f.write(f"Cleaned columns: {len(df.columns)}\n")
    f.write(f"Average rating: {df['rate'].mean():.3f}\n")
    f.write(f"Median rating: {df['rate'].median():.3f}\n")
    f.write(f"Missing review texts: {df['review'].isna().sum()}\n")
    f.write(f"Flagged long reviews: {df['review_length_outlier'].sum()}\n")
    f.write("\nRating counts:\n")
    f.write(df["rate"].value_counts().sort_index().to_string())
    f.write("\n\nSentiment counts:\n")
    f.write(df["sentiment"].value_counts(dropna=False).to_string())
    f.write("\n\nRating by sentiment:\n")
    f.write(sentiment_summary.to_string())

print("\nEDA charts and summary saved successfully!")
print("Next: inspect the generated results before drawing conclusions.")
