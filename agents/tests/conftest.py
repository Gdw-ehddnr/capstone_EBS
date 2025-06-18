# conftest.py
import pytest
from fastapi.testclient import TestClient
import sys
import os
from datetime import datetime, timedelta
from typing import List, Dict, Any
from agents.technical.models import MarketData
import uuid

# Add the project root directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

@pytest.fixture
def mock_market_data():
    """테스트용 시장 데이터"""
    return {
        "ticker": "TEST",
        "prices": [100.0 + i * 0.1 for i in range(100)],
        "volumes": [1000000 + i * 1000 for i in range(100)]
    }

def generate_test_market_data() -> List[MarketData]:
    """테스트용 시장 데이터 생성"""
    base_time = datetime.now()
    data = []
    for i in range(50):
        timestamp = (base_time - timedelta(days=i)).isoformat()
        data.append(MarketData(
            ticker="AAPL",
            timestamp=timestamp,
            open=100.0 + i,
            high=102.0 + i,
            low=98.0 + i,
            close=101.0 + i,
            volume=1000000 + i * 1000
        ))
    return data