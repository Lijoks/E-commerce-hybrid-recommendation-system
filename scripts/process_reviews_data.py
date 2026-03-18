import json
import pandas as pd
import numpy as np
from pathlib import Path
import logging
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import gc  # Garbage collection

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ChunkedReviewProcessor:
    def __init__(self, chunk_size=50000):
        self.reviews_file = Path("data/raw/reviews/Electronics.jsonl")
        self.products_df = None
        self.chunk_size = chunk_size  # Process 50k reviews at a time
        self.all_interactions = []    # Store paths to chunk files, not data

    def load_product_catalog(self):
        """Load your product catalog (this one is fine to load fully)."""
        logger.info("=" * 60)
        logger.info("Loading Product Catalog")
        logger.info("=" * 60)
        self.products_df = pd.read_csv('data/raw/amazon_products.csv')
        logger.info(f"Loaded {len(self.products_df):,} products")
        return self.products_df

    def process_in_chunks(self, max_chunks=None):
        """Process the large JSONL file in chunks to manage memory."""
        logger.info("=" * 60)
        logger.info("Processing Reviews in Chunks")
        logger.info("=" * 60)

        chunk_count = 0
        current_chunk = []
        output_dir = Path('data/processed/chunks')
        output_dir.mkdir(parents=True, exist_ok=True)

        # Open the file and read line by line (memory efficient)
        with open(self.reviews_file, 'r', encoding='utf-8') as f:
            # Use tqdm for a progress bar (no need to know total lines)
            for line in tqdm(f, desc="Processing reviews line by line"):
                try:
                    review = json.loads(line.strip())
                    current_chunk.append({
                        'user_id': review.get('user_id'),
                        'asin': review.get('parent_asin'),
                        'rating': review.get('rating'),
                        'timestamp': review.get('timestamp'),
                    })

                    # When chunk is full, process and save it
                    if len(current_chunk) >= self.chunk_size:
                        self._save_chunk(current_chunk, chunk_count, output_dir)
                        current_chunk = []  # Clear memory
                        chunk_count += 1
                        gc.collect()  # Force garbage collection

                        if max_chunks and chunk_count >= max_chunks:
                            break

                except json.JSONDecodeError:
                    logger.warning("Skipping invalid JSON line")
                    continue

        # Save the final partial chunk
        if current_chunk:
            self._save_chunk(current_chunk, chunk_count, output_dir)
            chunk_count += 1

        logger.info(f"✅ Created {chunk_count} chunk files in {output_dir}")
        return chunk_count

    def _save_chunk(self, chunk_data, chunk_index, output_dir):
        """Save a single chunk to disk and clear it from memory."""
        chunk_df = pd.DataFrame(chunk_data)

        # Join with product catalog for this chunk
        merged_chunk = pd.merge(
            chunk_df,
            self.products_df[['asin']],  # Only need asin for validation
            on='asin',
            how='inner'
        )

        if len(merged_chunk) > 0:
            # Save to parquet (efficient, compressed format)
            chunk_file = output_dir / f'chunk_{chunk_index:04d}.parquet'
            merged_chunk.to_parquet(chunk_file, index=False)

    def combine_and_split(self, max_chunks=None):
        """Combine all chunk files and create train/test split."""
        logger.info("=" * 60)
        logger.info("Combining Chunks and Creating Train/Test Split")
        logger.info("=" * 60)

        chunk_dir = Path('data/processed/chunks')
        all_chunks = sorted(chunk_dir.glob('chunk_*.parquet'))

        if max_chunks:
            all_chunks = all_chunks[:max_chunks]

        logger.info(f"Found {len(all_chunks)} chunk files to combine")

        # Combine chunks progressively to manage memory
        combined_df = pd.DataFrame()
        for i, chunk_file in enumerate(all_chunks):
            logger.info(f"Loading chunk {i+1}/{len(all_chunks)}")
            chunk_df = pd.read_parquet(chunk_file)

            # Create interaction label
            chunk_df['interaction'] = (chunk_df['rating'] >= 4).astype(int)

            combined_df = pd.concat([combined_df, chunk_df], ignore_index=True)

            # Optional: if combined_df gets too big, we could save intermediate results
            # but for final split we need it all

        logger.info(f"Total combined data: {len(combined_df):,} rows")
        logger.info(f"Unique users: {combined_df['user_id'].nunique():,}")
        logger.info(f"Unique products: {combined_df['asin'].nunique():,}")

        # Create train/test split
        user_counts = combined_df['user_id'].value_counts()
        active_users = user_counts[user_counts >= 5].index.tolist()
        logger.info(f"Users with 5+ interactions: {len(active_users)}")

        if len(active_users) < 10:
            active_users = combined_df['user_id'].unique()

        train_users, test_users = train_test_split(
            active_users, test_size=0.2, random_state=42
        )

        train_df = combined_df[combined_df['user_id'].isin(train_users)]
        test_df = combined_df[combined_df['user_id'].isin(test_users)]

        logger.info(f"Train: {len(train_df):,} interactions")
        logger.info(f"Test: {len(test_df):,} interactions")

        # Save final datasets
        output_dir = Path('data/processed')
        train_df.to_parquet(output_dir / 'train_interactions.parquet', index=False)
        test_df.to_parquet(output_dir / 'test_interactions.parquet', index=False)

        # Clean up chunk files to save disk space (optional)
        # for chunk_file in all_chunks:
        #     chunk_file.unlink()

        logger.info(f"✅ Final datasets saved to {output_dir}")
        return train_df, test_df

def main():
    processor = ChunkedReviewProcessor(chunk_size=50000)

    # Load product catalog
    processor.load_product_catalog()

    # Process reviews in chunks (start with max_chunks=5 for testing)
    # processor.process_in_chunks(max_chunks=5)  # Test with 5 chunks first
    processor.process_in_chunks()  # Process all chunks when ready

    # Combine and create train/test split (start with max_chunks=5 for testing)
    # processor.combine_and_split(max_chunks=5)
    processor.combine_and_split()

if __name__ == "__main__":
    main()