import pandas as pd
import numpy as np
from pathlib import Path
import pickle
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RecommenderTester:
    """Simple tester for the hybrid recommender."""
    
    def __init__(self):
        self.processed_dir = Path('data/processed')
        self.products_df = pd.read_csv('data/raw/amazon_products.csv')
        
    def load_models(self):
        """Load both models."""
        # Load CF model
        with open(self.processed_dir / 'cf_model.pkl', 'rb') as f:
            cf_data = pickle.load(f)
        
        self.cf_model = cf_data['model']
        self.user_to_idx = cf_data['user_to_idx']
        self.item_to_idx = cf_data['item_to_idx']
        self.idx_to_item = cf_data['idx_to_item']
        
        # Load reranker if available
        reranker_path = self.processed_dir / 'reranker_model.pkl'
        if reranker_path.exists():
            with open(reranker_path, 'rb') as f:
                reranker_data = pickle.load(f)
            self.reranker = reranker_data['model']
            self.has_reranker = True
            logger.info("Reranker loaded")
        else:
            self.has_reranker = False
            logger.warning("No reranker found, using CF only")
    
    def get_product_info(self, asin):
        """Get product title and info."""
        product = self.products_df[self.products_df['asin'] == asin]
        if len(product) > 0:
            return product.iloc[0]['title']
        return "Unknown Product"
    
    def recommend_for_user(self, user_id, n_recommendations=10):
        """Generate recommendations for a user."""
        if user_id not in self.user_to_idx:
            logger.error(f"User {user_id} not found")
            return []
        
        user_idx = self.user_to_idx[user_id]
        
        # Get recommendations from ALS
        item_indices, scores = self.cf_model.recommend(
            user_idx,
            self.cf_model.item_user_data,
            N=n_recommendations,
            filter_already_liked_items=True
        )
        
        # Convert to item IDs
        recommendations = []
        for idx, score in zip(item_indices, scores):
            asin = self.idx_to_item[idx]
            title = self.get_product_info(asin)
            recommendations.append({
                'asin': asin,
                'title': title,
                'score': float(score)
            })
        
        return recommendations
    
    def test_sample_users(self, n_users=5):
        """Test with sample users."""
        logger.info("=" * 60)
        logger.info("Testing Recommender with Sample Users")
        logger.info("=" * 60)
        
        # Get some sample users
        train_df = pd.read_parquet(self.processed_dir / 'train_interactions.parquet')
        sample_users = train_df['user_id'].unique()[:n_users]
        
        for user_id in sample_users:
            recommendations = self.recommend_for_user(user_id, n_recommendations=5)
            
            logger.info(f"\n📱 User {user_id}:")
            for i, rec in enumerate(recommendations, 1):
                logger.info(f"  {i}. {rec['title'][:80]}... (score: {rec['score']:.3f})")

def main():
    tester = RecommenderTester()
    tester.load_models()
    tester.test_sample_users(n_users=5)

if __name__ == "__main__":
    main()