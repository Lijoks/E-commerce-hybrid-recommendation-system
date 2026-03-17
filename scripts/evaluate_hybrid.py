# scripts/evaluate_hybrid_fixed.py
import pandas as pd
import numpy as np
from pathlib import Path
import pickle
import logging
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class HybridEvaluator:
    """
    Evaluate hybrid recommender performance.
    """
    
    def __init__(self):
        self.processed_dir = Path('data/processed')
        self.cf_model = None
        self.reranker = None
        self.scaler = None
        self.feature_names = None
        self.user_to_idx = None
        self.item_to_idx = None
        self.idx_to_item = None
        self.products_df = None
        self.category_map = None
        
    def load_models(self):
        """Load both CF and hybrid models."""
        logger.info("=" * 60)
        logger.info("Loading Models")
        logger.info("=" * 60)
        
        # Load CF model
        cf_path = self.processed_dir / 'cf_model.pkl'
        if not cf_path.exists():
            logger.error(f"CF model not found at {cf_path}")
            return False
            
        with open(cf_path, 'rb') as f:
            cf_data = pickle.load(f)
        
        self.cf_model = cf_data['model']
        self.user_to_idx = cf_data['user_to_idx']
        self.item_to_idx = cf_data['item_to_idx']
        self.idx_to_item = {v: k for k, v in self.item_to_idx.items()}
        
        logger.info(f"CF Model loaded with {self.cf_model.factors} factors")
        logger.info(f"Users in model: {len(self.user_to_idx):,}")
        logger.info(f"Items in model: {len(self.item_to_idx):,}")
        
        # Load reranker model
        reranker_path = self.processed_dir / 'full_reranker_model.pkl'
        if reranker_path.exists():
            with open(reranker_path, 'rb') as f:
                reranker_data = pickle.load(f)
            self.reranker = reranker_data['model']
            self.scaler = reranker_data['scaler']
            self.feature_names = reranker_data['feature_names']
            logger.info("✅ Reranker model loaded")
        else:
            logger.warning("No reranker model found - will evaluate CF only")
        
        # Load product data for features
        self.products_df = pd.read_csv('data/raw/amazon_products.csv')
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
        
        self.category_map = dict(zip(self.products_df['asin'], self.products_df['category_id']))
        self.product_features = self.products_df.set_index('asin')
        
        return True
    
    def get_cf_recommendations(self, user_id, n=100):
        """Get CF recommendations for a user using dot product (no matrix needed)."""
        if user_id not in self.user_to_idx:
            return []
        
        user_idx = self.user_to_idx[user_id]
        
        try:
            # Calculate scores for all items using dot product
            user_factors = self.cf_model.user_factors[user_idx]
            item_factors = self.cf_model.item_factors
            
            # Calculate dot product for all items
            scores = np.dot(item_factors, user_factors)
            
            # Get top N items
            top_indices = np.argsort(scores)[-n:][::-1]
            
            recommendations = [self.idx_to_item[idx] for idx in top_indices]
            return recommendations
            
        except Exception as e:
            logger.error(f"Error getting CF recommendations: {e}")
            return []
    
    def get_hybrid_score(self, user_id, item_id, train_df):
        """Get hybrid score for a user-item pair."""
        if self.reranker is None:
            return 0.0
        
        try:
            # Get CF score
            cf_score = self.get_cf_score(user_id, item_id)
            
            # Get item features
            if item_id in self.product_features.index:
                product = self.product_features.loc[item_id]
                
                # Get user features
                user_history = train_df[train_df['user_id'] == user_id]
                user_avg_rating = user_history['rating'].mean() if len(user_history) > 0 else 3.0
                
                # Category match
                item_category = self.category_map.get(item_id, -1)
                user_categories = [
                    self.category_map.get(asin) 
                    for asin in user_history['asin'].tolist() 
                    if asin in self.category_map
                ]
                category_match = 1.0 if item_category in user_categories else 0.0
                
                # Create feature vector
                features = np.array([[
                    cf_score,
                    float(product['popularity_score']),
                    float(product['log_price']),
                    float(product['log_reviews']),
                    float(product['log_bought']),
                    float(product['stars']),
                    int(product['isBestSeller']),
                    float(user_avg_rating),
                    category_match
                ]])
                
                # Scale and predict
                features_scaled = self.scaler.transform(features)
                score = self.reranker.predict(features_scaled)[0]
                return score
            else:
                return cf_score
                
        except Exception as e:
            logger.debug(f"Error getting hybrid score: {e}")
            return 0.0
    
    def get_cf_score(self, user_id, item_id):
        """Get CF score for a user-item pair."""
        if user_id not in self.user_to_idx or item_id not in self.item_to_idx:
            return 0.0
        
        user_idx = self.user_to_idx[user_id]
        item_idx = self.item_to_idx[item_id]
        
        user_factors = self.cf_model.user_factors[user_idx]
        item_factors = self.cf_model.item_factors[item_idx]
        
        return float(np.dot(user_factors, item_factors))
    
    def precision_at_k(self, recommended, relevant, k):
        """Precision@K."""
        if len(recommended) > k:
            recommended = recommended[:k]
        hits = len(set(recommended) & set(relevant))
        return hits / k if k > 0 else 0
    
    def recall_at_k(self, recommended, relevant, k):
        """Recall@K."""
        if len(recommended) > k:
            recommended = recommended[:k]
        hits = len(set(recommended) & set(relevant))
        return hits / len(relevant) if len(relevant) > 0 else 0
    
    def ndcg_at_k(self, recommended, relevant, k):
        """NDCG@K."""
        if len(recommended) > k:
            recommended = recommended[:k]
        
        relevance = [1 if item in relevant else 0 for item in recommended]
        dcg = sum(rel / np.log2(i + 2) for i, rel in enumerate(relevance))
        
        ideal_relevance = sorted([1] * min(len(relevant), k) + [0] * (k - min(len(relevant), k)), reverse=True)
        idcg = sum(rel / np.log2(i + 2) for i, rel in enumerate(ideal_relevance))
        
        return dcg / idcg if idcg > 0 else 0
    
    def evaluate(self, n_users=100):
        """Evaluate both CF and hybrid models."""
        logger.info("=" * 60)
        logger.info("Evaluating Models")
        logger.info("=" * 60)
        
        # Load test data
        test_path = self.processed_dir / 'test_interactions.parquet'
        train_path = self.processed_dir / 'train_interactions.parquet'
        
        if not test_path.exists():
            logger.error(f"Test data not found at {test_path}")
            return None
            
        test_df = pd.read_parquet(test_path)
        train_df = pd.read_parquet(train_path)
        
        logger.info(f"Loaded {len(test_df):,} test interactions")
        logger.info(f"Test users: {test_df['user_id'].nunique():,}")
        
        # Get users in both test and model
        test_users = [u for u in test_df['user_id'].unique() if u in self.user_to_idx]
        logger.info(f"Users in both test and model: {len(test_users):,}")
        
        # Sample users for evaluation
        if len(test_users) > n_users:
            np.random.seed(42)
            test_users = np.random.choice(test_users, n_users, replace=False)
        
        logger.info(f"Evaluating on {len(test_users)} users...")
        
        metrics = defaultdict(list)
        
        for user_id in test_users:
            # Get user's test items
            user_test_items = set(
                test_df[test_df['user_id'] == user_id]['asin'].tolist()
            )
            
            if len(user_test_items) == 0:
                continue
            
            # Get all items user has interacted with (to filter out)
            user_history = set(train_df[train_df['user_id'] == user_id]['asin'].tolist())
            
            # Generate candidates from all items
            all_items = list(self.item_to_idx.keys())
            candidates = [item for item in all_items if item not in user_history]
            
            if len(candidates) < 20:
                continue
            
            # Score candidates with CF
            cf_scores = [(item, self.get_cf_score(user_id, item)) for item in candidates[:200]]
            cf_scores.sort(key=lambda x: x[1], reverse=True)
            cf_recommendations = [item for item, _ in cf_scores]
            
            # Score candidates with hybrid if available
            if self.reranker is not None:
                hybrid_scores = [(item, self.get_hybrid_score(user_id, item, train_df)) 
                                for item in candidates[:200]]
                hybrid_scores.sort(key=lambda x: x[1], reverse=True)
                hybrid_recommendations = [item for item, _ in hybrid_scores]
            else:
                hybrid_recommendations = cf_recommendations
            
            # Calculate metrics for different K values
            for k in [5, 10, 20]:
                metrics[f'cf_precision@{k}'].append(
                    self.precision_at_k(cf_recommendations, user_test_items, k)
                )
                metrics[f'cf_recall@{k}'].append(
                    self.recall_at_k(cf_recommendations, user_test_items, k)
                )
                metrics[f'cf_ndcg@{k}'].append(
                    self.ndcg_at_k(cf_recommendations, user_test_items, k)
                )
                
                metrics[f'hybrid_precision@{k}'].append(
                    self.precision_at_k(hybrid_recommendations, user_test_items, k)
                )
                metrics[f'hybrid_recall@{k}'].append(
                    self.recall_at_k(hybrid_recommendations, user_test_items, k)
                )
                metrics[f'hybrid_ndcg@{k}'].append(
                    self.ndcg_at_k(hybrid_recommendations, user_test_items, k)
                )
        
        # Average metrics
        results = {}
        for metric, values in metrics.items():
            if values:
                results[metric] = np.mean(values)
            else:
                results[metric] = 0.0
        
        # Display results
        logger.info("\n" + "=" * 60)
        logger.info("Evaluation Results")
        logger.info("=" * 60)
        
        for k in [5, 10, 20]:
            cf_prec = results.get(f'cf_precision@{k}', 0)
            cf_rec = results.get(f'cf_recall@{k}', 0)
            cf_ndcg = results.get(f'cf_ndcg@{k}', 0)
            hybrid_prec = results.get(f'hybrid_precision@{k}', 0)
            hybrid_rec = results.get(f'hybrid_recall@{k}', 0)
            hybrid_ndcg = results.get(f'hybrid_ndcg@{k}', 0)
            
            logger.info(f"\n@{k}:")
            logger.info(f"  CF - Precision: {cf_prec:.4f}, Recall: {cf_rec:.4f}, NDCG: {cf_ndcg:.4f}")
            logger.info(f"  Hybrid - Precision: {hybrid_prec:.4f}, Recall: {hybrid_rec:.4f}, NDCG: {hybrid_ndcg:.4f}")
        
        return results

def main():
    evaluator = HybridEvaluator()
    
    # Load models
    if not evaluator.load_models():
        logger.error("Failed to load models")
        return
    
    # Evaluate
    results = evaluator.evaluate(n_users=100)
    
    if results:
        logger.info("\n✅ Evaluation complete!")
    else:
        logger.error("Evaluation failed - no results")

if __name__ == "__main__":
    main()