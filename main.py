# ================================
# Exploratory Data Analysis
# ================================

# 1. Import libraries
import pandas as pd
import matplotlib.pyplot as plt

# ----------------------
# 2. Load Datasets
# ----------------------
brands = pd.read_csv("brands.csv")
finance = pd.read_csv("finance.csv")
info = pd.read_csv("info.csv")
reviews = pd.read_csv("reviews.csv")

# ----------------------
# 3. Merge DataFrames on 'product_id'
# ----------------------
merged_df = info.merge(finance, on="product_id") \
                .merge(reviews, on="product_id") \
                .merge(brands, on="product_id")

# Drop rows with missing values
merged_df.dropna(subset=["listing_price", "revenue", "brand"], inplace=True)

# ----------------------
# 4. Price Labeling (quartiles)
# ----------------------
price_labels = ["Budget", "Average", "Expensive", "Elite"]
merged_df["price_label"] = pd.qcut(
    merged_df["listing_price"],
    q=4,
    labels=price_labels
)

# Ensure ordering
merged_df["price_label"] = pd.Categorical(merged_df["price_label"], categories=price_labels, ordered=True)

# ----------------------
# 5. Brand & Price Aggregation
# ----------------------
adidas_vs_nike = merged_df.groupby(["brand", "price_label"], as_index=False).agg(
    num_products=("price_label", "count"),
    mean_revenue=("revenue", "mean")
).round(2)

# ----------------------
# 6. Description Length Analysis
# ----------------------
# Keep numeric length for reference
merged_df["description_len_num"] = merged_df["description"].str.len()

# Define bins and labels
length_bins = [0, 100, 200, 300, 400, 500, 600, 700]
length_labels = ["100", "200", "300", "400", "500", "600", "700"]

# Categorize description lengths
merged_df["description_length"] = pd.cut(
    merged_df["description_len_num"],
    bins=length_bins,
    labels=length_labels
)

# Aggregate by description length bins
description_lengths = merged_df.groupby("description_length", as_index=False).agg(
    mean_rating=("rating", "mean"),
    total_reviews=("reviews", "sum")
).round(2)

# ----------------------
# 7. Pivot tables for dashboard
# ----------------------
pivot_products = adidas_vs_nike.pivot(
    index="price_label",
    columns="brand",
    values="num_products"
)

pivot_revenue = adidas_vs_nike.pivot(
    index="price_label",
    columns="brand",
    values="mean_revenue"
)

# ----------------------
# 8. Dashboard: 2x2 Subplots
# ----------------------
fig, axes = plt.subplots(2, 2, figsize=(12, 8))

# Chart 1: Products by Price Category
pivot_products.plot(kind="bar", ax=axes[0, 0])
axes[0, 0].set_title("Products by Price Category")
axes[0, 0].set_xlabel("Price Category")
axes[0, 0].set_ylabel("Number of Products")
axes[0, 0].set_xticklabels(pivot_products.index, rotation=0)

# Chart 2: Revenue by Price Category
pivot_revenue.plot(kind="bar", ax=axes[0, 1])
axes[0, 1].set_title("Revenue by Price Category")
axes[0, 1].set_xlabel("Price Category")
axes[0, 1].set_ylabel("Average Revenue")
axes[0, 1].set_xticklabels(pivot_revenue.index, rotation=0)

# Chart 3: Rating vs Description Length
axes[1, 0].plot(
    description_lengths["description_length"],
    description_lengths["mean_rating"],
    marker='o'
)
axes[1, 0].set_title("Rating vs Description Length")
axes[1, 0].set_xlabel("Description Length (bins)")
axes[1, 0].set_ylabel("Average Rating")

# Chart 4: Reviews vs Description Length
axes[1, 1].bar(
    description_lengths["description_length"],
    description_lengths["total_reviews"]
)
axes[1, 1].set_title("Reviews vs Description Length")
axes[1, 1].set_xlabel("Description Length (bins)")
axes[1, 1].set_ylabel("Total Reviews")

plt.tight_layout()
plt.savefig("dashboard.png", dpi=300)  # Save high-res
plt.show()

