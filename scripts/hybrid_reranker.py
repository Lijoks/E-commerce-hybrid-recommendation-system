# scripts/hybrid_reranker_final_fixed.py - With correct data path
import pandas as pd
import numpy as np
from pathlib import Path
import pickle
import logging
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from scipy import sparse
import warnings
warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FullHybridReranker:
    """
    Hybrid recommender with full-scale training.
    """
    
    def __init__(self):
        self.processed_dir = Path('data/processed')
        self.raw_data_dir = Path('data/raw')  # Add raw data directory
        self.cf_model = None
        self.user_to_idx = None
        self.item_to_idx = None
        self.idx_to_item = None
        self.products_df = None
        self.lgb_model = None
        self.scaler = StandardScaler()
        # Initialize feature names here
        self.feature_names = [
            'cf_score', 'popularity', 'price', 'reviews', 'bought',
            'stars', 'is_best_seller', 'user_avg_rating', 'category_match'
        ]
        
    def load_models(self):
        """Load pre-trained collaborative filtering model."""
        logger.info("=" * 60)
        logger.info("Loading Collaborative Filtering Model")
        logger.info("=" * 60)
        
        model_path = self.processed_dir / 'cf_model.pkl'
        if not model_path.exists():
            logger.error(f"CF model not found at {model_path}")
            return False
        
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)
        
        self.cf_model = model_data['model']
        self.user_to_idx = model_data['user_to_idx']
        self.item_to_idx = model_data['item_to_idx']
        self.idx_to_item = {v: k for k, v in self.item_to_idx.items()}
        
        logger.info(f"Model loaded with {self.cf_model.factors} factors")
        logger.info(f"Users in model: {len(self.user_to_idx):,}")
        logger.info(f"Items in model: {len(self.item_to_idx):,}")
        return True
        
    def load_product_features(self):
        """Load product catalog with engineered features."""
        logger.info("=" * 60)
        logger.info("Loading Product Features")
        logger.info("=" * 60)
        
        # Use the correct path: data/raw/amazon_products.csv
        product_path = self.raw_data_dir / 'amazon_products.csv'
        
        if not product_path.exists():
            logger.error(f"Product catalog not found at {product_path}")
            return False
        
        self.products_df = pd.read_csv(product_path)
        logger.info(f"Loaded {len(self.products_df):,} products from {product_path}")
        logger.info(f"Product columns: {self.products_df.columns.tolist()}")
        
        # Basic features
        self.products_df['log_price'] = np.log1p(self.products_df['price'].fillna(0))
        self.products_df['log_reviews'] = np.log1p(self.products_df['reviews'].fillna(0))
        self.products_df['log_bought'] = np.log1p(self.products_df['boughtInLastMonth'].fillna(0))
        
        # Popularity score
        max_reviews = self.products_df['log_reviews'].max()
        max_bought = self.products_df['log_bought'].max()
        
        self.products_df['popularity_score'] = (
            self.products_df['stars'].fillna(3) * 0.4 +
            (self.products_df['log_reviews'] / max_reviews if max_reviews > 0 else 0) * 0.3 +
            (self.products_df['log_bought'] / max_bought if max_bought > 0 else 0) * 0.3
        )
        
        # Create category mapping
        self.category_map = dict(zip(self.products_df['asin'], self.products_df['category_id']))
        self.product_features = self.products_df.set_index('asin')
        
        logger.info(f"Product features ready: {len(self.products_df):,} items")
        logger.info(f"Sample categories: {list(self.category_map.values())[:5]}")
        
        return True
        
    def get_cf_score(self, user_id, item_id):
        """Get CF score for a user-item pair."""
        if user_id not in self.user_to_idx or item_id not in self.item_to_idx:
            return 0.0
        
        user_idx = self.user_to_idx[user_id]
        item_idx = self.item_to_idx[item_id]
        
        user_factors = self.cf_model.user_factors[user_idx]
        item_factors = self.cf_model.item_factors[item_idx]
        
        return float(np.dot(user_factors, item_factors))
    
    def extract_features(self, item_id, user_id, train_df):
        """Extract features for a single item-user pair."""
        features = {}
        
        # Item features
        if item_id in self.product_features.index:
            product = self.product_features.loc[item_id]
            features['popularity'] = float(product['popularity_score'])
            features['price'] = float(product['log_price'])
            features['reviews'] = float(product['log_reviews'])
            features['bought'] = float(product['log_bought'])
            features['stars'] = float(product['stars'])
            features['is_best_seller'] = int(product['isBestSeller'])
        else:
            features['popularity'] = 0.0
            features['price'] = 0.0
            features['reviews'] = 0.0
            features['bought'] = 0.0
            features['stars'] = 0.0
            features['is_best_seller'] = 0
        
        # User features
        user_history = train_df[train_df['user_id'] == user_id]
        features['user_avg_rating'] = float(user_history['rating'].mean()) if len(user_history) > 0 else 3.0
        
        # Category match
        item_category = self.category_map.get(item_id, -1)
        user_categories = [
            self.category_map.get(asin) 
            for asin in user_history['asin'].tolist() 
            if asin in self.category_map
        ]
        features['category_match'] = 1.0 if item_category in user_categories else 0.0
        
        # Return as list in correct order (excluding cf_score)
        return [
            features['popularity'],
            features['price'],
            features['reviews'],
            features['bought'],
            features['stars'],
            features['is_best_seller'],
            features['user_avg_rating'],
            features['category_match']
        ]
    
    def prepare_training_data(self, train_df, n_users=3000, samples_per_user=20):
        """
        Prepare large-scale training data.
        """
        logger.info("=" * 60)
        logger.info("Preparing Large-Scale Training Data")
        logger.info("=" * 60)
        
        # Create interaction column if it doesn't exist
        if 'interaction' not in train_df.columns:
            logger.info("Creating interaction column from ratings (rating >= 4)")
            train_df['interaction'] = (train_df['rating'] >= 4).astype(int)
        
        # Get users with enough interactions
        user_counts = train_df['user_id'].value_counts()
        active_users = user_counts[user_counts >= 10].index.tolist()
        
        if len(active_users) > n_users:
            np.random.seed(42)
            active_users = np.random.choice(active_users, n_users, replace=False)
        
        logger.info(f"Selected {len(active_users):,} users with 10+ interactions")
        
        X_list = []
        y_list = []
        
        # Get all products for negative sampling
        all_products = list(self.item_to_idx.keys())
        logger.info(f"Total products available for sampling: {len(all_products):,}")
        
        for user_id in tqdm(active_users, desc="Preparing training data"):
            try:
                # Get user's positive items (rating >= 4)
                user_positives = train_df[
                    (train_df['user_id'] == user_id) & 
                    (train_df['interaction'] == 1)
                ]['asin'].tolist()
                
                if len(user_positives) < 5:
                    continue
                
                # Sample positives
                n_pos = min(8, len(user_positives))
                pos_samples = np.random.choice(user_positives, n_pos, replace=False)
                
                # Sample negatives (items user hasn't interacted with)
                user_negatives = [p for p in all_products if p not in user_positives]
                n_neg = n_pos * 2  # 2:1 negative to positive ratio
                
                if len(user_negatives) < n_neg:
                    continue
                    
                neg_samples = np.random.choice(user_negatives, n_neg, replace=False)
                
                # Create feature vectors for each sample
                for item_id in pos_samples:
                    cf_score = self.get_cf_score(user_id, item_id)
                    other_features = self.extract_features(item_id, user_id, train_df)
                    # Combine cf_score with other features
                    feature_vector = [cf_score] + other_features
                    X_list.append(feature_vector)
                    y_list.append(1)
                
                for item_id in neg_samples:
                    cf_score = self.get_cf_score(user_id, item_id)
                    other_features = self.extract_features(item_id, user_id, train_df)
                    feature_vector = [cf_score] + other_features
                    X_list.append(feature_vector)
                    y_list.append(0)
                    
            except Exception as e:
                logger.warning(f"Error processing user {user_id}: {e}")
                continue
        
        if not X_list:
            logger.error("No training data generated!")
            return None, None
        
        X = np.array(X_list)
        y = np.array(y_list)
        
        logger.info(f"Created {len(X):,} training samples")
        logger.info(f"Positive samples: {sum(y):,} ({sum(y)/len(y):.1%})")
        logger.info(f"Negative samples: {len(y)-sum(y):,}")
        logger.info(f"Feature dimension: {X.shape[1]}")
        
        return X, y
    
    def train_reranker(self, train_df):
        """Train LightGBM reranker with full data."""
        logger.info("=" * 60)
        logger.info("Training Full-Scale LightGBM Reranker")
        logger.info("=" * 60)
        
        # Prepare training data
        X, y = self.prepare_training_data(train_df, n_users=3000, samples_per_user=20)
        
        if X is None:
            logger.error("Failed to prepare training data")
            return None
        
        # Split into train and validation
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
        
        # Create LightGBM dataset
        train_data = lgb.Dataset(
            X_train_scaled, 
            label=y_train,
            feature_name=self.feature_names
        )
        val_data = lgb.Dataset(
            X_val_scaled, 
            label=y_val,
            reference=train_data
        )
        
        # Parameters
        params = {
            'objective': 'binary',
            'metric': 'auc',
            'boosting_type': 'gbdt',
            'num_leaves': 31,
            'learning_rate': 0.05,
            'feature_fraction': 0.8,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'lambda_l1': 0.1,
            'lambda_l2': 0.1,
            'min_data_in_leaf': 20,
            'verbose': -1,
            'random_state': 42,
            'n_jobs': -1
        }
        
        # Train model
        self.lgb_model = lgb.train(
            params,
            train_data,
            valid_sets=[val_data],
            num_boost_round=100,
            callbacks=[
                lgb.early_stopping(10),
                lgb.log_evaluation(10)
            ]
        )
        
        logger.info("Full-scale training complete!")
        
        # Feature importance
        importance = pd.DataFrame({
            'feature': self.feature_names,
            'importance': self.lgb_model.feature_importance()
        }).sort_values('importance', ascending=False)
        
        logger.info("\nFeature Importance:")
        for _, row in importance.iterrows():
            logger.info(f"  {row['feature']:20}: {row['importance']}")
        
        return self.lgb_model
    
    def save_model(self, filepath):
        """Save model and scaler."""
        model_data = {
            'model': self.lgb_model,
            'scaler': self.scaler,
            'feature_names': self.feature_names
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        logger.info(f"Model saved to {filepath}")

def main():
    # Initialize
    reranker = FullHybridReranker()
    
    # Load models and data
    if not reranker.load_models():
        logger.error("Failed to load CF model")
        return
    
    if not reranker.load_product_features():
        logger.error("Failed to load product features")
        return
    
    # Load training data
    train_path = reranker.processed_dir / 'train_interactions.parquet'
    if not train_path.exists():
        logger.error(f"Training data not found at {train_path}")
        return
    
    train_df = pd.read_parquet(train_path)
    logger.info(f"Loaded {len(train_df):,} training interactions")
    logger.info(f"Training users: {train_df['user_id'].nunique():,}")
    logger.info(f"Training items: {train_df['asin'].nunique():,}")
    
    # Train
    reranker.train_reranker(train_df)
    
    # Save
    reranker.save_model(reranker.processed_dir / 'full_reranker_model.pkl')

if __name__ == "__main__":
    main()