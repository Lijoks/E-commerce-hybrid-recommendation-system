# scripts/train_collaborative_filtering_fixed.py
import pandas as pd
import numpy as np
from pathlib import Path
import pickle
import logging
import implicit
from scipy import sparse
from implicit.nearest_neighbours import bm25_weight

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CollaborativeFiltering:
    def __init__(self, factors=50, iterations=15, regularization=0.01):
        self.factors = factors
        self.iterations = iterations
        self.regularization = regularization
        self.model = None
        self.user_to_idx = {}
        self.item_to_idx = {}
        self.idx_to_user = {}
        self.idx_to_item = {}
        
    def prepare_data(self, interactions_df):
        """Prepare data for ALS model."""
        logger.info("Preparing data for ALS...")
        
        # Create interaction column if it doesn't exist
        if 'interaction' not in interactions_df.columns:
            logger.info("Creating interaction column from ratings (rating >= 4)")
            interactions_df['interaction'] = (interactions_df['rating'] >= 4).astype(int)
        
        # Filter out zero interactions to save memory
        interactions_df = interactions_df[interactions_df['interaction'] > 0]
        logger.info(f"Positive interactions: {len(interactions_df):,}")
        
        # Create mappings
        unique_users = interactions_df['user_id'].unique()
        unique_items = interactions_df['asin'].unique()
        
        self.user_to_idx = {user: idx for idx, user in enumerate(unique_users)}
        self.item_to_idx = {item: idx for idx, item in enumerate(unique_items)}
        self.idx_to_user = {idx: user for user, idx in self.user_to_idx.items()}
        self.idx_to_item = {idx: item for item, idx in self.item_to_idx.items()}
        
        # Create sparse matrix
        user_indices = interactions_df['user_id'].map(self.user_to_idx)
        item_indices = interactions_df['asin'].map(self.item_to_idx)
        data = interactions_df['interaction'].values
        
        sparse_matrix = sparse.csr_matrix(
            (data, (user_indices, item_indices)),
            shape=(len(unique_users), len(unique_items))
        )
        
        # Apply BM25 weighting
        weighted_matrix = bm25_weight(sparse_matrix, K1=100, B=0.8)
        
        logger.info(f"Sparse matrix shape: {weighted_matrix.shape}")
        logger.info(f"Non-zero entries: {weighted_matrix.nnz:,}")
        logger.info(f"Users: {len(unique_users):,}")
        logger.info(f"Items: {len(unique_items):,}")
        
        return weighted_matrix
    
    def train(self, interactions_df):
        """Train ALS model."""
        logger.info(f"Training ALS with {self.factors} factors...")
        
        # Prepare data
        sparse_matrix = self.prepare_data(interactions_df)
        
        # Initialize model
        self.model = implicit.als.AlternatingLeastSquares(
            factors=self.factors,
            iterations=self.iterations,
            regularization=self.regularization,
            random_state=42,
            use_gpu=False
        )
        
        # Train model
        self.model.fit(sparse_matrix)
        
        logger.info("Model training complete!")
        
        return self.model
    
    def save_model(self, filepath):
        """Save model and mappings."""
        model_data = {
            'model': self.model,
            'user_to_idx': self.user_to_idx,
            'item_to_idx': self.item_to_idx,
            'idx_to_user': self.idx_to_user,
            'idx_to_item': self.idx_to_item,
            'user_factors': self.model.user_factors,
            'item_factors': self.model.item_factors
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        logger.info(f"Model saved to {filepath}")

def main():
    # Load training data
    processed_dir = Path('data/processed')
    train_path = processed_dir / 'train_interactions.parquet'
    
    if not train_path.exists():
        logger.error(f"Training data not found at {train_path}")
        return
    
    train_df = pd.read_parquet(train_path)
    logger.info(f"Training data: {len(train_df):,} interactions")
    logger.info(f"Users: {train_df['user_id'].nunique():,}")
    logger.info(f"Items: {train_df['asin'].nunique():,}")
    logger.info(f"Rating distribution:\n{train_df['rating'].value_counts().sort_index()}")
    
    # Train model
    cf = CollaborativeFiltering(factors=50, iterations=15)
    cf.train(train_df)
    
    # Save model
    cf.save_model(processed_dir / 'cf_model.pkl')

if __name__ == "__main__":
    main()