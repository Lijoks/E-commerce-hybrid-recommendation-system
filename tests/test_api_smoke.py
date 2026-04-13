# tests/test_api_smoke.py
"""
Smoke tests for the API (without actually starting the server).
"""

import pytest
from pathlib import Path

class TestAPIConfiguration:
    """Test API configuration."""
    
    def test_app_file_exists(self):
        """Test that app.py exists."""
        app_path = Path("app.py")
        assert app_path.exists(), "app.py not found in root directory"
    
    def test_app_has_required_imports(self):
        """Test that app.py has required imports."""
        with open("app.py", "r", encoding="utf-8") as f:
            content = f.read()
        
        required_imports = [
            "from fastapi import FastAPI",
            "import uvicorn",
            "import pandas as pd",
            "import numpy as np"
        ]
        
        for imp in required_imports:
            assert imp in content, f"Missing import: {imp}"
    
    def test_app_has_required_endpoints(self):
        """Test that app.py has required endpoints."""
        with open("app.py", "r", encoding="utf-8") as f:
            content = f.read()
        
        required_endpoints = [
            "@app.get('/health')",
            "@app.post('/recommend')",
            "@app.get('/stats')"
        ]
        
        for endpoint in required_endpoints:
            assert endpoint in content, f"Missing endpoint: {endpoint}"