"""
Pytest configuration file with fixtures for testing.
"""
import os
import pytest
import asyncio
from fastapi.testclient import TestClient
import sys
import logging
from unittest.mock import patch

# Add the parent directory to the path so we can import the app
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app.main import app


@pytest.fixture
def test_client():
    """
    Create a FastAPI test client for the app.
    """
    with TestClient(app) as client:
        yield client


@pytest.fixture(scope="session")
def event_loop():
    """
    Create an instance of the default event loop for each test case.
    This prevents "Event loop is closed" errors in async tests.
    """
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def mock_litellm():
    """
    Mock LiteLLM for testing.
    """
    with patch('app.main.litellm') as mock:
        yield mock


@pytest.fixture
def mock_ollama_client():
    """
    Mock Ollama client for testing.
    """
    with patch('ollama.Client') as mock:
        mock_instance = mock.return_value
        yield mock_instance
