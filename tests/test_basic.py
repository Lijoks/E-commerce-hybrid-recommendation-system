# tests/test_basic.py - Updated version
"""
Basic tests for the recommender system.
"""

import pytest
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

class TestImports:
    """Test that all required imports work."""
    
    def test_pandas_import(self):
        import pandas as pd
        assert pd.__version__ is not None
    
    def test_numpy_import(self):
        import numpy as np
        assert np.__version__ is not None
    
    def test_fastapi_import(self):
        import fastapi
        assert fastapi.__version__ is not None
    
    def test_lightgbm_import(self):
        import lightgbm as lgb
        assert lgb.__version__ is not None
    
    def test_implicit_import(self):
        import implicit
        assert implicit.__version__ is not None

class TestFileStructure:
    """Test that essential files exist."""
    
    def test_app_exists(self):
        assert Path("app.py").exists()
    
    def test_requirements_exists(self):
        assert Path("requirements.txt").exists()
    
    def test_readme_exists(self):
        assert Path("README.md").exists()
    
    def test_data_processed_exists(self):
        assert Path("data/processed").exists()
    
    def test_data_raw_exists(self):
        assert Path("data/raw").exists()

class TestModelFiles:
    """Test that trained model files exist."""
    
    def test_cf_model_exists(self):
        cf_path = Path("data/processed/cf_model.pkl")
        if cf_path.exists():
            assert cf_path.stat().st_size > 0
        else:
            pytest.skip("CF model not found - run training first")
    
    def test_reranker_model_exists(self):
        reranker_path = Path("data/processed/full_reranker_model.pkl")
        if reranker_path.exists():
            assert reranker_path.stat().st_size > 0
        else:
            pytest.skip("Reranker model not found - run training first")
    
    def test_product_data_exists(self):
        product_path = Path("data/raw/amazon_products.csv")
        if product_path.exists():
            assert product_path.stat().st_size > 0
        else:
            pytest.skip("Product data not found")

class TestAPIEndpoints:
    """Test API endpoint definitions (without running server)."""
    
    def test_health_endpoint_defined(self):
        with open("app.py", "r") as f:
            content = f.read()
        # Check for both single and double quotes
        assert ("@app.get('/health')" in content or 
                '@app.get("/health")' in content or
                "def health_check" in content)
    
    def test_recommend_endpoint_defined(self):
        with open("app.py", "r") as f:
            content = f.read()
        # Check for both single and double quotes
        assert ("@app.post('/recommend')" in content or 
                '@app.post("/recommend")' in content or
                "def get_recommendations" in content)