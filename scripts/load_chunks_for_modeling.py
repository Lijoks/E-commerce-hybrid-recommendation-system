import pandas as pd
import numpy as np
from pathlib import Path
import logging
from tqdm import tqdm
import pickle

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataLoader:
    def __init__(self):
        self.processed_dir = Path('data/processed')
        self.chunks_dir = self.processed_dir / 'chunks'
        self.products_df = None
        
    def load_product_catalog(self):
        """Load the product catalog."""
        logger.info("Loading product catalog...")
        self.products_df = pd.read_csv('data/raw/amazon_products.csv')
        logger.info(f"Loaded {len(self.products_df):,} products")
        return self.products_df
    
    def load_all_interactions(self, sample_size=None):
        """Load and combine interaction chunks."""
        logger.info("Loading interaction chunks...")
        
        chunk_files = sorted(self.chunks_dir.glob('chunk_*.parquet'))
        logger.info(f"Found {len(chunk_files)} chunk files")
        
        if sample_size:
            chunk_files = chunk_files[:sample_size]
            logger.info(f"Using {len(chunk_files)} chunks for sampling")
        
        all_interactions = []
        total_rows = 0
        
        for chunk_file in tqdm(chunk_files, desc="Loading chunks"):
            chunk_df = pd.read_parquet(chunk_file)
            all_interactions.append(chunk_df)
            total_rows += len(chunk_df)
            
            # Optional: print progress
            if len(all_interactions) % 10 == 0:
                logger.info(f"Loaded {len(all_interactions)} chunks ({total_rows:,} rows)")
        
        # Combine all chunks
        interactions_df = pd.concat(all_interactions, ignore_index=True)
        
        # Create interaction label if not present
        if 'interaction' not in interactions_df.columns:
            interactions_df['interaction'] = (interactions_df['rating'] >= 4).astype(int)
        
        logger.info(f"Total interactions: {len(interactions_df):,}")
        logger.info(f"Unique users: {interactions_df['user_id'].nunique():,}")
        logger.info(f"Unique products: {interactions_df['asin'].nunique():,}")
        logger.info(f"Rating distribution:\n{interactions_df['rating'].value_counts().sort_index()}")
        
        return interactions_df
    
    def create_user_item_matrix(self, interactions_df, min_interactions=5):
        """Create user-item matrix for collaborative filtering."""
        logger.info("Creating user-item matrix...")
        
        # Filter users with minimum interactions
        user_counts = interactions_df['user_id'].value_counts()
        active_users = user_counts[user_counts >= min_interactions].index
        interactions_filtered = interactions_df[interactions_df['user_id'].isin(active_users)]
        
        logger.info(f"Users with {min_interactions}+ interactions: {len(active_users):,}")
        logger.info(f"Interactions after filtering: {len(interactions_filtered):,}")
        
        # Create pivot table
        user_item_matrix = interactions_filtered.pivot_table(
            index='user_id',
            columns='asin',
            values='interaction',
            fill_value=0
        ).astype(np.int8)
        
        logger.info(f"User-item matrix shape: {user_item_matrix.shape}")
        logger.info(f"Sparsity: {1 - (user_item_matrix > 0).sum().sum() / user_item_matrix.size:.4%}")
        
        return user_item_matrix
    
    def create_train_test_split(self, interactions_df, test_size=0.2):
        """Create train/test split by user."""
        from sklearn.model_selection import train_test_split
        
        logger.info("Creating train/test split...")
        
        unique_users = interactions_df['user_id'].unique()
        train_users, test_users = train_test_split(
            unique_users, test_size=test_size, random_state=42
        )
        
        train_df = interactions_df[interactions_df['user_id'].isin(train_users)]
        test_df = interactions_df[interactions_df['user_id'].isin(test_users)]
        
        logger.info(f"Train: {len(train_df):,} interactions ({len(train_users):,} users)")
        logger.info(f"Test: {len(test_df):,} interactions ({len(test_users):,} users)")
        
        # Save splits
        train_df.to_parquet(self.processed_dir / 'train_interactions.parquet', index=False)
        test_df.to_parquet(self.processed_dir / 'test_interactions.parquet', index=False)
        
        return train_df, test_df

def main():
    loader = DataLoader()
    
    # Load product catalog
    products_df = loader.load_product_catalog()
    
    # Load all interactions (use sample_size=5 for testing, remove for full)
    interactions_df = loader.load_all_interactions()  # Add sample_size=5 for testing
    
    # Create train/test split
    train_df, test_df = loader.create_train_test_split(interactions_df)
    
    # Create user-item matrix
    user_item_matrix = loader.create_user_item_matrix(train_df)
    
    # Save matrix for later use
    with open(loader.processed_dir / 'user_item_matrix.pkl', 'wb') as f:
        pickle.dump(user_item_matrix, f)
    
    logger.info("✅ Data preparation complete!")
    logger.info(f"Files saved in {loader.processed_dir}")

if __name__ == "__main__":
    main()