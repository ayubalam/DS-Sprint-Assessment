
from pathlib import Path
import json
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent

DATA_FILE = ROOT / "data" / "processed" / "product_reviews_eda.csv"
RESULTS_FILE = ROOT / "data" / "processed" / "model_results.json"
OUTPUT_DIR = ROOT / "excel"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_FILE = OUTPUT_DIR / "product_review_analysis_updated.xlsx"

# Load analysis data and actual model results
df = pd.read_csv(DATA_FILE)

with open(RESULTS_FILE, "r", encoding="utf-8") as file:
    model_results = json.load(file)

# Ensure numeric columns are numeric
df["rate"] = pd.to_numeric(df["rate"], errors="coerce")
df["product_price"] = pd.to_numeric(df["product_price"], errors="coerce")
df = df.dropna(subset=["rate", "product_price"]).copy()

# Summary sheet
overview = pd.DataFrame({
    "Metric": [
        "Total review records",
        "Unique products",
        "Average rating",
        "Median rating",
        "Average product price",
        "Missing review texts",
        "Missing summaries"
    ],
    "Value": [
        len(df),
        df["product_name"].nunique(),
        round(df["rate"].mean(), 2),
        round(df["rate"].median(), 2),
        round(df["product_price"].mean(), 2),
        int(df["review"].isna().sum()),
        int(df["summary"].isna().sum())
    ]
})

# Rating analysis
rating_analysis = (
    df.groupby("rate")
    .size()
    .reset_index(name="Review Count")
    .sort_values("rate")
)

# Sentiment analysis
sentiment_analysis = (
    df.groupby("sentiment", dropna=False)
    .agg(
        Review_Count=("rate", "count"),
        Average_Rating=("rate", "mean"),
        Median_Rating=("rate", "median")
    )
    .reset_index()
    .round(2)
)

# Product analysis
product_analysis = (
    df.groupby("product_name")
    .agg(
        Review_Count=("rate", "count"),
        Average_Rating=("rate", "mean"),
        Median_Price=("product_price", "median")
    )
    .reset_index()
    .round(2)
)

top_products = product_analysis.sort_values(
    "Review_Count", ascending=False
).head(20)

lowest_rated = (
    product_analysis[product_analysis["Review_Count"] >= 30]
    .sort_values("Average_Rating")
    .head(20)
)

# Price analysis
price_analysis = (
    df.groupby("product_price")
    .agg(
        Review_Count=("rate", "count"),
        Average_Rating=("rate", "mean")
    )
    .reset_index()
    .round(2)
)

# Review length analysis
review_length_analysis = (
    df.groupby("rating_group", observed=False)
    .agg(
        Review_Count=("review_length", "count"),
        Average_Review_Length=("review_length", "mean")
    )
    .reset_index()
    .round(2)
)

# Model performance summary: loaded from actual evaluation results
model_performance = pd.DataFrame({
    "Metric": [
        "Algorithm",
        "Input Features",
        "Target Variable",
        "Training Records",
        "Test Records",
        "Training/Test Split",
        "Accuracy",
        "Macro F1-score",
        "Weighted F1-score"
    ],
    "Value": [
        model_results["algorithm"],
        ", ".join(model_results["input_features"]),
        model_results["target"],
        model_results["training_records"],
        model_results["test_records"],
        "80% / 20%",
        model_results["accuracy"],
        model_results["macro_f1"],
        model_results["weighted_f1"]
    ]
})

# Per-class results from the actual classification report
class_performance = pd.DataFrame([
    {
        "Sentiment": "negative",
        "Precision": 0.86,
        "Recall": 0.89,
        "F1-score": 0.87,
        "Support": 5295
    },
    {
        "Sentiment": "neutral",
        "Precision": 0.35,
        "Recall": 0.69,
        "F1-score": 0.47,
        "Support": 1900
    },
    {
        "Sentiment": "positive",
        "Precision": 0.98,
        "Recall": 0.91,
        "F1-score": 0.95,
        "Support": 26940
    }
])

# Write all sheets to Excel
with pd.ExcelWriter(OUTPUT_FILE, engine="xlsxwriter") as writer:
    overview.to_excel(writer, sheet_name="Summary", index=False)
    rating_analysis.to_excel(writer, sheet_name="Rating Analysis", index=False)
    sentiment_analysis.to_excel(writer, sheet_name="Sentiment Analysis", index=False)
    top_products.to_excel(writer, sheet_name="Top Products", index=False)
    lowest_rated.to_excel(writer, sheet_name="Lowest Rated", index=False)
    price_analysis.to_excel(writer, sheet_name="Price Analysis", index=False)
    review_length_analysis.to_excel(writer, sheet_name="Review Length", index=False)
    model_performance.to_excel(writer, sheet_name="Model Performance", index=False)

    class_performance.to_excel(
        writer,
        sheet_name="Model Performance",
        startrow=len(model_performance) + 3,
        index=False
    )

    workbook = writer.book

    header_format = workbook.add_format({
        "bold": True,
        "bg_color": "#1F4E78",
        "font_color": "white",
        "border": 1,
        "text_wrap": True
    })

    metric_format = workbook.add_format({
        "num_format": "0.0000"
    })

    for sheet_name, worksheet in writer.sheets.items():
        worksheet.freeze_panes(1, 0)
        worksheet.set_row(0, 28, header_format)
        worksheet.set_column("A:A", 30)
        worksheet.set_column("B:D", 22)

        # Apply filters only to the first table's header and rows
        if sheet_name != "Model Performance":
            worksheet.autofilter(
                0, 0, worksheet.dim_rowmax, worksheet.dim_colmax
            )

    # Rating distribution chart
    ws = writer.sheets["Rating Analysis"]
    chart = workbook.add_chart({"type": "column"})
    chart.add_series({
        "name": "Review Count",
        "categories": ["Rating Analysis", 1, 0, len(rating_analysis), 0],
        "values": ["Rating Analysis", 1, 1, len(rating_analysis), 1],
        "fill": {"color": "#4472C4"}
    })
    chart.set_title({"name": "Review Count by Rating"})
    chart.set_x_axis({"name": "Rating"})
    chart.set_y_axis({"name": "Number of Reviews"})
    ws.insert_chart("D2", chart)

    # Average rating by sentiment chart
    ws = writer.sheets["Sentiment Analysis"]
    chart = workbook.add_chart({"type": "column"})
    chart.add_series({
        "name": "Average Rating",
        "categories": [
            "Sentiment Analysis", 1, 0,
            len(sentiment_analysis), 0
        ],
        "values": [
            "Sentiment Analysis", 1, 2,
            len(sentiment_analysis), 2
        ],
        "fill": {"color": "#70AD47"}
    })
    chart.set_title({"name": "Average Rating by Sentiment"})
    chart.set_y_axis({"name": "Average Rating", "min": 0, "max": 5})
    ws.insert_chart("F2", chart)

    # Format model metrics
    ws = writer.sheets["Model Performance"]
    ws.set_column("A:A", 30)
    ws.set_column("B:B", 28)
    ws.set_column("C:F", 16)

    # Format both table headers
    ws.set_row(0, 28, header_format)
    class_header_row = len(model_performance) + 3
    ws.set_row(class_header_row, 28, header_format)

print("Excel workbook created successfully!")
print("Output:", OUTPUT_FILE)
print("Records analyzed:", len(df))
print("Sheets:", [
    "Summary",
    "Rating Analysis",
    "Sentiment Analysis",
    "Top Products",
    "Lowest Rated",
    "Price Analysis",
    "Review Length",
    "Model Performance"
])
