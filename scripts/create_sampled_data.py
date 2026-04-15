import pandas as pd

# Load full product data
df = pd.read_csv('data/raw/amazon_products.csv')

# Take only 500 products for Vercel demo
sampled_df = df.head(500)

# Save as new file
sampled_df.to_csv('data/raw/amazon_products_sampled.csv', index=False)

print(f"✅ Created sampled file with {len(sampled_df)} products")
print(f"   Original: {len(df):,} products")