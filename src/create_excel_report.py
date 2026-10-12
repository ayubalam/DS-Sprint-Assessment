
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA_FILE = ROOT / "data" / "processed" / "product_reviews_eda.csv"
OUTPUT_DIR = ROOT / "excel"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_FILE = OUTPUT_DIR / "product_review_analysis.xlsx"

df = pd.read_csv(DATA_FILE)

# Ensure numeric columns are numeric
df["rate"] = pd.to_numeric(df["rate"], errors="coerce")
df["product_price"] = pd.to_numeric(df["product_price"], errors="coerce")
df = df.dropna(subset=["rate", "product_price"]).copy()

# Summary calculations from the actual dataset
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

rating_analysis = (
    df.groupby("rate")
    .size()
    .reset_index(name="Review Count")
    .sort_values("rate")
)

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

price_analysis = (
    df.groupby("product_price")
    .agg(
        Review_Count=("rate", "count"),
        Average_Rating=("rate", "mean")
    )
    .reset_index()
    .round(2)
)

review_length_analysis = (
    df.groupby("rating_group", observed=False)
    .agg(
        Review_Count=("review_length", "count"),
        Average_Review_Length=("review_length", "mean")
    )
    .reset_index()
    .round(2)
)

# Write sheets and create Excel charts
with pd.ExcelWriter(OUTPUT_FILE, engine="xlsxwriter") as writer:
    overview.to_excel(writer, sheet_name="Summary", index=False)
    rating_analysis.to_excel(writer, sheet_name="Rating Analysis", index=False)
    sentiment_analysis.to_excel(writer, sheet_name="Sentiment Analysis", index=False)
    top_products.to_excel(writer, sheet_name="Top Products", index=False)
    lowest_rated.to_excel(writer, sheet_name="Lowest Rated", index=False)
    price_analysis.to_excel(writer, sheet_name="Price Analysis", index=False)
    review_length_analysis.to_excel(
        writer, sheet_name="Review Length", index=False
    )

    workbook = writer.book

    header_format = workbook.add_format({
        "bold": True,
        "bg_color": "#1F4E78",
        "font_color": "white",
        "border": 1,
        "text_wrap": True
    })

    for sheet_name, worksheet in writer.sheets.items():
        worksheet.freeze_panes(1, 0)
        worksheet.set_row(0, 28, header_format)
        worksheet.set_column("A:A", 34)
        worksheet.set_column("B:D", 22)
        worksheet.autofilter(0, 0, worksheet.dim_rowmax, worksheet.dim_colmax)

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

    # Sentiment average rating chart
    ws = writer.sheets["Sentiment Analysis"]
    chart = workbook.add_chart({"type": "column"})
    chart.add_series({
        "name": "Average Rating",
        "categories": [
            "Sentiment Analysis", 1, 0, len(sentiment_analysis), 0
        ],
        "values": [
            "Sentiment Analysis", 1, 2, len(sentiment_analysis), 2
        ],
        "fill": {"color": "#70AD47"}
    })
    chart.set_title({"name": "Average Rating by Sentiment"})
    chart.set_y_axis({"name": "Average Rating", "min": 0, "max": 5})
    ws.insert_chart("F2", chart)

print("Excel workbook created successfully!")
print("Output:", OUTPUT_FILE)
print("Records analyzed:", len(df))
print("Sheets:", [
    "Summary", "Rating Analysis", "Sentiment Analysis",
    "Top Products", "Lowest Rated", "Price Analysis", "Review Length"
])
