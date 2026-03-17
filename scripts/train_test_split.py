# scripts/32_fix_train_test_split.py
import pandas as pd
import numpy as np
from pathlib import Path
import logging
from sklearn.model_selection import train_test_split
import pickle
import shutil

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TrainTestFixer:
    """
    Fix train/test split to ensure user overlap between train and test.
    """
    
    def __init__(self):
        self.processed_dir = Path('data/processed')
        self.raw_reviews_dir = Path('data/raw/reviews')
        
    def check_current_split(self):
        """Check the current train/test split."""
        logger.info("=" * 60)
        logger.info("Checking Current Train/Test Split")
        logger.info("=" * 60)
        
        train_path = self.processed_dir / 'train_interactions.parquet'
        test_path = self.processed_dir / 'test_interactions.parquet'
        
        if not train_path.exists() or not test_path.exists():
            logger.error("Train/test files not found")
            return False
        
        train_df = pd.read_parquet(train_path)
        test_df = pd.read_parquet(test_path)
        
        logger.info(f"Train: {len(train_df):,} interactions")
        logger.info(f"Train users: {train_df['user_id'].nunique():,}")
        logger.info(f"Test: {len(test_df):,} interactions")
        logger.info(f"Test users: {test_df['user_id'].nunique():,}")
        
        # Check overlap
        train_users = set(train_df['user_id'].unique())
        test_users = set(test_df['user_id'].unique())
        overlap = train_users & test_users
        
        logger.info(f"Users in both train and test: {len(overlap)}")
        
        if len(overlap) == 0:
            logger.warning("⚠️ No user overlap between train and test!")
            logger.warning("This will make evaluation impossible")
            return False
        
        logger.info(f"✅ User overlap: {len(overlap):,} users")
        return True
    
    def create_proper_split(self, interactions_file='data/processed/all_interactions.parquet', 
                           test_size=0.2, min_interactions=5):
        """
        Create a proper train/test split with user overlap.
        """
        logger.info("=" * 60)
        logger.info("Creating Proper Train/Test Split")
        logger.info("=" * 60)
        
        # Load all interactions
        if Path(interactions_file).exists():
            all_df = pd.read_parquet(interactions_file)
        else:
            # Try to load from chunks
            logger.info("Loading from chunks...")
            chunk_files = list((self.processed_dir / 'chunks').glob('*.parquet'))
            if not chunk_files:
                logger.error("No interaction data found!")
                return None
            
            chunks = []
            for chunk_file in chunk_files:
                chunks.append(pd.read_parquet(chunk_file))
            all_df = pd.concat(chunks, ignore_index=True)
        
        logger.info(f"Total interactions: {len(all_df):,}")
        logger.info(f"Total users: {all_df['user_id'].nunique():,}")
        logger.info(f"Total items: {all_df['asin'].nunique():,}")
        
        # Filter users with enough interactions
        user_counts = all_df['user_id'].value_counts()
        active_users = user_counts[user_counts >= min_interactions].index
        all_df = all_df[all_df['user_id'].isin(active_users)]
        
        logger.info(f"After filtering (min {min_interactions} interactions):")
        logger.info(f"  Interactions: {len(all_df):,}")
        logger.info(f"  Users: {all_df['user_id'].nunique():,}")
        
        # Get unique users
        users = all_df['user_id'].unique()
        
        # Split users into train and test
        train_users, test_users = train_test_split(
            users, test_size=test_size, random_state=42
        )
        
        logger.info(f"Train users: {len(train_users):,}")
        logger.info(f"Test users: {len(test_users):,}")
        
        # Create train and test dataframes
        train_df = all_df[all_df['user_id'].isin(train_users)]
        test_df = all_df[all_df['user_id'].isin(test_users)]
        
        # For each test user, keep only some interactions for testing
        # This is leave-one-out style evaluation
        test_interactions = []
        train_interactions = []
        
        for user_id in test_users:
            user_data = test_df[test_df['user_id'] == user_id]
            
            # Get positive interactions (rating >= 4)
            positive = user_data[user_data['rating'] >= 4]
            
            if len(positive) >= 2:
                # Use one positive for testing, rest for training
                test_idx = np.random.choice(positive.index, size=1)
                test_interactions.append(user_data.loc[test_idx])
                
                # Rest go to training
                train_idx = user_data.index.difference(test_idx)
                train_interactions.append(user_data.loc[train_idx])
            else:
                # User has few positives, use all for training
                train_interactions.append(user_data)
        
        # Combine
        final_train = pd.concat([train_df] + train_interactions, ignore_index=True)
        final_test = pd.concat(test_interactions, ignore_index=True) if test_interactions else pd.DataFrame()
        
        logger.info(f"Final train: {len(final_train):,} interactions")
        logger.info(f"Final test: {len(final_test):,} interactions")
        
        # Verify overlap
        final_train_users = set(final_train['user_id'].unique())
        final_test_users = set(final_test['user_id'].unique())
        overlap = final_train_users & final_test_users
        
        logger.info(f"Users in both final train and test: {len(overlap)}")
        
        if len(overlap) == 0:
            logger.error("❌ Still no overlap! Something went wrong.")
            return None
        
        logger.info(f"✅ Success! {len(overlap):,} users shared between train and test")
        
        return final_train, final_test
    
    def save_split(self, train_df, test_df, backup=True):
        """Save the new train/test split."""
        logger.info("=" * 60)
        logger.info("Saving New Train/Test Split")
        logger.info("=" * 60)
        
        # Backup old files if they exist
        if backup:
            for file in ['train_interactions.parquet', 'test_interactions.parquet']:
                old_path = self.processed_dir / file
                if old_path.exists():
                    backup_path = self.processed_dir / f"{file}.backup"
                    shutil.copy2(old_path, backup_path)
                    logger.info(f"Backed up {file} to {backup_path}")
        
        # Save new files
        train_path = self.processed_dir / 'train_interactions.parquet'
        test_path = self.processed_dir / 'test_interactions.parquet'
        
        train_df.to_parquet(train_path, index=False)
        test_df.to_parquet(test_path, index=False)
        
        logger.info(f"Saved train: {len(train_df):,} interactions to {train_path}")
        logger.info(f"Saved test: {len(test_df):,} interactions to {test_path}")
        
        # Save statistics
        stats = {
            'train_interactions': len(train_df),
            'train_users': train_df['user_id'].nunique(),
            'train_items': train_df['asin'].nunique(),
            'test_interactions': len(test_df),
            'test_users': test_df['user_id'].nunique(),
            'test_items': test_df['asin'].nunique(),
            'shared_users': len(set(train_df['user_id'].unique()) & set(test_df['user_id'].unique()))
        }
        
        stats_path = self.processed_dir / 'split_statistics.txt'
        with open(stats_path, 'w') as f:
            f.write("Train/Test Split Statistics\n")
            f.write("=" * 40 + "\n")
            for key, value in stats.items():
                f.write(f"{key}: {value:,}\n")
        
        logger.info(f"Statistics saved to {stats_path}")
        
    def verify_new_split(self):
        """Verify the new split."""
        logger.info("=" * 60)
        logger.info("Verifying New Split")
        logger.info("=" * 60)
        
        train_df = pd.read_parquet(self.processed_dir / 'train_interactions.parquet')
        test_df = pd.read_parquet(self.processed_dir / 'test_interactions.parquet')
        
        train_users = set(train_df['user_id'].unique())
        test_users = set(test_df['user_id'].unique())
        overlap = train_users & test_users
        
        logger.info(f"Train users: {len(train_users):,}")
        logger.info(f"Test users: {len(test_users):,}")
        logger.info(f"Overlapping users: {len(overlap):,}")
        
        if len(overlap) > 0:
            logger.info(f"✅ Verification passed! {len(overlap)} users in both sets")
            
            # Show sample user
            sample_user = list(overlap)[0]
            train_samples = train_df[train_df['user_id'] == sample_user]
            test_samples = test_df[test_df['user_id'] == sample_user]
            
            logger.info(f"\nSample user {sample_user[:20]}...:")
            logger.info(f"  Train interactions: {len(train_samples)}")
            logger.info(f"  Test interactions: {len(test_samples)}")
            logger.info(f"  Train ratings: {train_samples['rating'].tolist()}")
            logger.info(f"  Test ratings: {test_samples['rating'].tolist()}")
            
            return True
        else:
            logger.error("❌ Verification failed - no overlapping users!")
            return False

def main():
    fixer = TrainTestFixer()
    
    # Check current split
    has_overlap = fixer.check_current_split()
    
    if not has_overlap:
        logger.warning("Current split has no user overlap. Creating new split...")
        
        # Create proper split
        result = fixer.create_proper_split(
            interactions_file='data/processed/all_interactions.parquet',
            test_size=0.2,
            min_interactions=5
        )
        
        if result:
            train_df, test_df = result
            fixer.save_split(train_df, test_df, backup=True)
            fixer.verify_new_split()
            
            logger.info("\n" + "=" * 60)
            logger.info("✅ Train/test split fixed successfully!")
            logger.info("=" * 60)
            logger.info("\nNext steps:")
            logger.info("1. Retrain your CF model with: python scripts/train_cf.py")
            logger.info("2. Retrain your reranker with: python scripts/hybrid_reranker.py")
            logger.info("3. Re-evaluate with: python scripts/evaluate_hybrid.py")
    else:
        logger.info("✅ Current split already has user overlap. No action needed.")

if __name__ == "__main__":
    main()