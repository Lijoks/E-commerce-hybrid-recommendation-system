# scripts/17_process_official_reviews.py
import json
import pandas as pd
import numpy as np
from pathlib import Path
import logging
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import gzip

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class OfficialReviewsProcessor:
    def __init__(self):
        self.reviews_file = Path("data/raw/reviews/Electronics.jsonl")
        self.meta_file = Path("data/raw/reviews/meta_Electronics.jsonl")
        self.products_df = None
        self.reviews_df = None
        self.meta_df = None
        self.joined_df = None
        
    def load_product_catalog(self):
        """Load your product catalog."""
        logger.info("=" * 60)
        logger.info("Loading Product Catalog")
        logger.info("=" * 60)
        
        self.products_df = pd.read_csv('data/raw/amazon_products.csv')
        logger.info(f"Loaded {len(self.products_df):,} products")
        logger.info(f"Columns: {self.products_df.columns.tolist()}")
        
        return self.products_df
    
    def load_reviews(self, max_reviews=None):
        """Load reviews from Electronics.jsonl."""
        logger.info("=" * 60)
        logger.info("Loading Reviews")
        logger.info("=" * 60)
        
        reviews = []
        
        # Check if file exists
        if not self.reviews_file.exists():
            logger.error(f"File not found: {self.reviews_file}")
            return None
        
        # Determine if gzipped
        open_func = gzip.open if str(self.reviews_file).endswith('.gz') else open
        mode = 'rt' if str(self.reviews_file).endswith('.gz') else 'r'
        
        with open_func(self.reviews_file, mode, encoding='utf-8') as f:
            for i, line in enumerate(tqdm(f, desc="Loading reviews")):
                if max_reviews and i >= max_reviews:
                    break
                try:
                    review = json.loads(line.strip())
                    reviews.append({
                        'user_id': review.get('reviewerID'),
                        'asin': review.get('asin'),
                        'rating': review.get('overall'),
                        'timestamp': review.get('unixReviewTime'),
                        'review_text': review.get('reviewText'),
                        'summary': review.get('summary'),
                        'helpful_votes': review.get('helpful', [0,0])[0] if review.get('helpful') else 0,
                        'total_votes': review.get('helpful', [0,0])[1] if review.get('helpful') else 0
                    })
                except json.JSONDecodeError as e:
                    logger.warning(f"Skipping invalid JSON at line {i}: {e}")
        
        self.reviews_df = pd.DataFrame(reviews)
        logger.info(f"Loaded {len(self.reviews_df):,} reviews")
        logger.info(f"Columns: {self.reviews_df.columns.tolist()}")
        logger.info(f"Rating distribution:\n{self.reviews_df['rating'].value_counts().sort_index()}")
        
        return self.reviews_df
    
    def load_metadata(self, max_items=None):
        """Load product metadata from meta_Electronics.jsonl."""
        logger.info("=" * 60)
        logger.info("Loading Product Metadata")
        logger.info("=" * 60)
        
        metadata = []
        
        if not self.meta_file.exists():
            logger.warning(f"Metadata file not found: {self.meta_file}")
            return None
        
        open_func = gzip.open if str(self.meta_file).endswith('.gz') else open
        mode = 'rt' if str(self.meta_file).endswith('.gz') else 'r'
        
        with open_func(self.meta_file, mode, encoding='utf-8') as f:
            for i, line in enumerate(tqdm(f, desc="Loading metadata")):
                if max_items and i >= max_items:
                    break
                try:
                    item = json.loads(line.strip())
                    metadata.append({
                        'asin': item.get('asin'),
                        'title': item.get('title'),
                        'description': item.get('description'),
                        'price': item.get('price'),
                        'brand': item.get('brand'),
                        'categories': '|'.join(item.get('categories', [[]])[0]) if item.get('categories') else None,
                        'also_bought': item.get('also_bought', []),
                        'image_url': item.get('imageURL', [None])[0] if item.get('imageURL') else None
                    })
                except json.JSONDecodeError as e:
                    logger.warning(f"Skipping invalid JSON at line {i}: {e}")
        
        self.meta_df = pd.DataFrame(metadata)
        logger.info(f"Loaded {len(self.meta_df):,} products from metadata")
        logger.info(f"Columns: {self.meta_df.columns.tolist()}")
        
        return self.meta_df
    
    def join_all_data(self):
        """Join reviews, metadata, and product catalog."""
        logger.info("=" * 60)
        logger.info("Joining All Data Sources")
        logger.info("=" * 60)
        
        # Step 1: Join reviews with your product catalog
        logger.info("Step 1: Joining reviews with product catalog...")
        reviews_with_products = pd.merge(
            self.reviews_df,
            self.products_df,
            on='asin',
            how='inner'
        )
        logger.info(f"  Result: {len(reviews_with_products):,} rows")
        
        # Step 2: Join with metadata if available
        if self.meta_df is not None:
            logger.info("Step 2: Joining with metadata...")
            self.joined_df = pd.merge(
                reviews_with_products,
                self.meta_df[['asin', 'title', 'description', 'brand', 'categories']],
                on='asin',
                how='left'
            )
            logger.info(f"  Result: {len(self.joined_df):,} rows")
        else:
            self.joined_df = reviews_with_products
        
        # Create interaction label
        self.joined_df['interaction'] = (self.joined_df['rating'] >= 4).astype(int)
        
        logger.info(f"\nFinal dataset:")
        logger.info(f"  Rows: {len(self.joined_df):,}")
        logger.info(f"  Users: {self.joined_df['user_id'].nunique():,}")
        logger.info(f"  Products: {self.joined_df['asin'].nunique():,}")
        logger.info(f"  Avg rating: {self.joined_df['rating'].mean():.2f}")
        logger.info(f"  Interaction rate: {self.joined_df['interaction'].mean():.2%}")
        
        return self.joined_df
    
    def create_features(self):
        """Create user and product features."""
        logger.info("=" * 60)
        logger.info("Creating Features")
        logger.info("=" * 60)
        
        # User features
        user_features = self.joined_df.groupby('user_id').agg({
            'rating': ['mean', 'std', 'count'],
            'interaction': 'mean',
            'timestamp': ['min', 'max']
        }).round(4)
        
        user_features.columns = ['_'.join(col).strip() for col in user_features.columns.values]
        user_features = user_features.reset_index()
        user_features['user_activity_days'] = (
            user_features['timestamp_max'] - user_features['timestamp_min']
        ) / (24*60*60)
        
        # Product features from your catalog
        product_features = self.products_df.copy()
        
        # Add review-based features
        product_stats = self.joined_df.groupby('asin').agg({
            'rating': ['mean', 'std', 'count'],
            'interaction': 'mean'
        }).round(4)
        product_stats.columns = ['_'.join(col).strip() for col in product_stats.columns.values]
        product_stats = product_stats.reset_index()
        
        product_features = pd.merge(
            product_features,
            product_stats,
            on='asin',
            how='left'
        )
        
        logger.info(f"User features: {user_features.shape}")
        logger.info(f"Product features: {product_features.shape}")
        
        return user_features, product_features
    
    def create_train_test_split(self):
        """Create train/test split."""
        logger.info("=" * 60)
        logger.info("Creating Train/Test Split")
        logger.info("=" * 60)
        
        # Split by user
        unique_users = self.joined_df['user_id'].unique()
        train_users, test_users = train_test_split(
            unique_users, test_size=0.2, random_state=42
        )
        
        train_df = self.joined_df[self.joined_df['user_id'].isin(train_users)]
        test_df = self.joined_df[self.joined_df['user_id'].isin(test_users)]
        
        logger.info(f"Train: {len(train_df):,} interactions ({len(train_users):,} users)")
        logger.info(f"Test: {len(test_df):,} interactions ({len(test_users):,} users)")
        
        return train_df, test_df
    
    def save_all_data(self, train_df, test_df, user_features, product_features):
        """Save all processed data."""
        logger.info("=" * 60)
        logger.info("Saving Processed Data")
        logger.info("=" * 60)
        
        output_dir = Path('data/processed')
        output_dir.mkdir(exist_ok=True)
        
        # Save interactions
        train_df.to_parquet(output_dir / 'train_interactions.parquet', index=False)
        test_df.to_parquet(output_dir / 'test_interactions.parquet', index=False)
        self.joined_df.to_parquet(output_dir / 'all_interactions.parquet', index=False)
        
        # Save features
        user_features.to_parquet(output_dir / 'user_features.parquet', index=False)
        product_features.to_parquet(output_dir / 'product_features.parquet', index=False)
        
        # Save summary
        with open(output_dir / 'dataset_summary.txt', 'w') as f:
            f.write("Amazon Electronics Reviews Dataset\n")
            f.write("=" * 40 + "\n")
            f.write(f"Total interactions: {len(self.joined_df):,}\n")
            f.write(f"Unique users: {self.joined_df['user_id'].nunique():,}\n")
            f.write(f"Unique products: {self.joined_df['asin'].nunique():,}\n")
            f.write(f"Rating range: {self.joined_df['rating'].min()} - {self.joined_df['rating'].max()}\n")
            f.write(f"Average rating: {self.joined_df['rating'].mean():.2f}\n")
            f.write(f"Interaction rate (rating>=4): {self.joined_df['interaction'].mean():.2%}\n")
            f.write(f"\nTrain size: {len(train_df):,}\n")
            f.write(f"Test size: {len(test_df):,}\n")
        
        logger.info(f"✅ All data saved to {output_dir}")
        
        # Show files
        for file in output_dir.glob("*"):
            if file.is_file():
                size_mb = file.stat().st_size / (1024 * 1024)
                logger.info(f"  - {file.name} ({size_mb:.2f} MB)")

def main():
    processor = OfficialReviewsProcessor()
    
    # Load product catalog
    processor.load_product_catalog()
    
    # Load reviews (start with 100k for testing, remove for full)
    processor.load_reviews(max_reviews=100000)
    
    # Load metadata (optional, but recommended)
    processor.load_metadata(max_items=50000)
    
    # Join all data
    processor.join_all_data()
    
    # Create features
    user_features, product_features = processor.create_features()
    
    # Create train/test split
    train_df, test_df = processor.create_train_test_split()
    
    # Save everything
    processor.save_all_data(train_df, test_df, user_features, product_features)
    
    print("\n" + "=" * 60)
    print("Sample of processed data:")
    print("=" * 60)
    print(processor.joined_df[['user_id', 'asin', 'rating', 'interaction']].head(10))

if __name__ == "__main__":
    main()