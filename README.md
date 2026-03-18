# 🛍️ E-commerce Hybrid Recommender System

[![Python](https://img.shields.io/badge/python-3.9%2B-blue)](https://www.python.org/)
[![LightGBM](https://img.shields.io/badge/LightGBM-4.1.0-green)](https://lightgbm.readthedocs.io/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104.1-teal)](https://fastapi.tiangolo.com/)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)

A production-ready hybrid recommender system combining collaborative filtering (ALS) with content-based features using LightGBM. Built with 1.4M Amazon products and real user interactions.

## 📊 Project Overview

This system demonstrates a complete ML pipeline from data processing to deployment:

- **Collaborative Filtering**: ALS algorithm from the `implicit` library
- **Content Features**: Product price, popularity, categories, and ratings
- **Hybrid Reranker**: LightGBM model combining both signals
- **Performance**: AUC of 0.97 on validation data

## 🏗️ Architecture
Raw Data → Feature Engineering → ALS → Candidate Generation → LightGBM Reranker → Recommendations
↓ ↓ ↓ ↓
Product Catalog User Profiles User-Item Matrix Final Rankings


## 📁 Project Structure
ecommerce-recommender/
├── src/ # Source code
│ ├── models/ # ML models (ALS, LightGBM)
│ └── utils/ # Helper functions
├── scripts/ # Executable scripts
│ ├── 01_process_data.py # Data processing pipeline
│ ├── 02_train_cf.py # Train collaborative filtering
│ ├── 03_hybrid_reranker.py # Train hybrid model
│ └── 04_evaluate.py # Model evaluation
├── tests/ # Unit tests
├── docker/ # Docker configuration
├── .github/workflows/ # CI/CD pipelines
└── requirements.txt # Dependencies


## 🚀 Quick Start

### Prerequisites
- Python 3.9+
- 16GB+ RAM (for full dataset)
- Git LFS (optional, for model files)

### Installation

```bash
# Clone the repository
git clone https://github.com/Lijoks/E-commerce-hybrid-recommendation-system
cd ecommerce-recommender

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

Data Setup

⚠️ Note: The dataset is too large for GitHub (21+ GB). See DATA_README.md for download instructions.

After downloading the data, run:
# Process raw data into training format
python scripts/process_reviews_data.py

# Train collaborative filtering model
python scripts/train_collaborative_filtering.py

# Load chunks for modeling
python scripts/load_chunks_for_modeling.py

# Split the data
python scripts/train_test_split.py

# Train hybrid reranker
python scripts/hybrid_reranker.py

# Evaluate model performance
python scripts/evaluate_hybrid.py


📈 Model Performance
Metric	Value
Training AUC	0.97
Users in Model	33,606
Products in Catalog	1.4M
Training Samples	68,796
🔧 Configuration

Create a .env file for custom settings:
env

DATA_DIR=./data
MODEL_DIR=./models
LOG_LEVEL=INFO

🐳 Docker Deployment
bash

# Build Docker image
docker build -t recommender-api -f docker/Dockerfile .

# Run container
docker run -p 8000:8000 recommender-api

🧪 Running Tests
bash

# Run all tests
pytest tests/

# Run with coverage
pytest tests/ --cov=src --cov-report=html

📚 API Documentation (Coming Soon)

Once the FastAPI service is running:

    Swagger UI: http://localhost:8000/docs

    ReDoc: http://localhost:8000/redoc

🤝 Contributing

    Fork the repository

    Create a feature branch (git checkout -b feature/amazing-feature)

    Commit changes (git commit -m 'Add amazing feature')

    Push to branch (git push origin feature/amazing-feature)

    Open a Pull Request

📝 License

This project is licensed under the MIT License - see the LICENSE file for details.
🙏 Acknowledgments

    McAuley Lab for the Amazon Reviews dataset

    Implicit library for ALS implementation

    LightGBM for gradient boosting

📧 Contact

Adedoyin Lijoka  lijoksadedoyin@gmail.com

Project Link: https://github.com/Lijoks/E-commerce-hybrid-recommendation-system
text


## 📝 **Create the DATA_README.md File**

```markdown
# DATA_README.md - Dataset Setup Instructions

## ⚠️ Important Note

The datasets used in this project are **too large for GitHub** (21+ GB total). This document explains how to obtain and set up the required data.

## 📦 Required Datasets

### 1. Amazon Products Dataset
- **File**: `amazon_products.csv`
- **Size**: 358 MB
- **Location**: `data/raw/amazon_products.csv`
- **Source**: [Kaggle - Amazon Products Dataset 2023](https://www.kaggle.com/datasets/your-dataset-link)

### 2. Amazon Electronics Reviews
- **Files**: 
  - `Electronics.jsonl` (21.5 GB)
  - `meta_Electronics.jsonl` (5 GB)
- **Location**: `data/raw/reviews/`
- **Source**: [HuggingFace - McAuley-Lab/Amazon-Reviews-2023](https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023)

## 📥 Download Instructions

### Option 1: Using Python (Recommended for Reviews)

```python
# Download reviews directly in Python
from datasets import load_dataset

# This will download the Electronics reviews
dataset = load_dataset(
    "McAuley-Lab/Amazon-Reviews-2023",
    "raw_review_Electronics",
    split="full"
)

# Save to the correct location
# The dataset will be cached automatically

Option 2: Manual Download

    Products Dataset: Download from Kaggle and place in data/raw/amazon_products.csv

    Reviews Dataset: Clone from HuggingFace or download via browser

📂 Expected Directory Structure

After downloading, your data/raw folder should look like this:
text

data/raw/
├── amazon_products.csv          # 358 MB
└── reviews/
    ├── Electronics.jsonl        # 21.5 GB
    └── meta_Electronics.jsonl   # 5 GB

🔄 Processing the Data

Once all files are in place, run:

# Process raw data into training format (creates chunks)
python scripts/process_reviews_data.py

# This will create processed files in data/processed/

📊 Processed Data Output

After running the processing script, you'll have:
text

data/processed/
├── chunks/                      # 50k-row chunks of interactions
├── train_interactions.parquet   # Training data (222k rows)
├── test_interactions.parquet    # Test data (6.6k rows)
├── cf_model.pkl                  # Trained ALS model (12 MB)
└── full_reranker_model.pkl       # Trained LightGBM model

⚡ Quick Test with Sample

If you want to test the pipeline with a smaller sample first:
python

# In scripts/01_process_data.py, modify the chunk loading:
processor.load_reviews(max_reviews=100000)  # Only load 100k reviews

❓ Troubleshooting
"File not found" errors

Ensure all files are in the correct locations as shown above.
Out of memory errors

The full dataset requires ~16GB RAM. If you have less, use the sampling approach above.
Slow processing

The first run will be slow as it processes large files. Subsequent runs will be faster due to caching.
📝 Notes

    The data files are listed in .gitignore and will not be committed to GitHub

    All scripts expect the data in the paths shown above

    Processed files can be regenerated at any time from the raw data
