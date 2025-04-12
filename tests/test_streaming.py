"""
Tests for the streaming functionality.
"""
import pytest
import json
from unittest.mock import MagicMock, patch
import asyncio
from fastapi.testclient import TestClient


@patch('app.main.litellm.completion')
def test_generate_streaming(mock_completion, test_client):
    """Test the generate endpoint with streaming."""
    # Create a mock for the streaming response
    class MockStreamingResponse:
        def __aiter__(self):
            return self
        
        async def __anext__(self):
            # Simulate stream end after 3 chunks
            if not hasattr(self, 'count'):
                self.count = 0
            
            if self.count >= 3:
                raise StopAsyncIteration
            
            chunk = MagicMock()
            chunk.choices = [MagicMock()]
            chunk.model = "ollama/llama3"
            chunk.created = 1713000000
            
            # Simulate different chunks of content
            contents = ["Hello", " world", "!"]
            chunk.choices[0].delta.content = contents[self.count]
            
            self.count += 1
            return chunk
    
    # Set up the mock to return our streaming response
    mock_completion.return_value = MockStreamingResponse()
    
    # Make a request to generate endpoint with streaming
    request_data = {
        "model": "llama3",
        "prompt": "Say hello",
        "stream": True
    }
    
    # Use the test client to make a streaming request
    response = test_client.post("/api/generate", json=request_data)
    
    # Check that the response is a streaming response
    assert response.status_code == 200
    assert response.headers["content-type"] == "text/event-stream"
    
    # Parse the streaming response
    chunks = list(response.iter_lines())
    assert len(chunks) >= 3  # We should have at least 3 chunks plus a final done message
    
    # Verify the content of each chunk
    all_content = ""
    for i, chunk in enumerate(chunks):
        if not chunk:  # Skip empty lines
            continue
        
        data = json.loads(chunk)
        if "done" in data and data["done"] is True:
            # This is the final chunk
            continue
        
        assert "model" in data
        assert "response" in data
        assert "done" in data
        assert data["done"] is False
        all_content += data["response"]
    
    # Check that we received the complete message
    assert "Hello world!" in all_content


@patch('app.main.litellm.completion')
def test_chat_streaming(mock_completion, test_client):
    """Test the chat endpoint with streaming."""
    # Create a mock for the streaming response
    class MockStreamingResponse:
        def __aiter__(self):
            return self
        
        async def __anext__(self):
            # Simulate stream end after 3 chunks
            if not hasattr(self, 'count'):
                self.count = 0
            
            if self.count >= 3:
                raise StopAsyncIteration
            
            chunk = MagicMock()
            chunk.choices = [MagicMock()]
            chunk.model = "ollama/llama3"
            chunk.created = 1713000000
            
            # Simulate different chunks of content
            contents = ["I'm ", "doing ", "well!"]
            chunk.choices[0].delta.content = contents[self.count]
            
            self.count += 1
            return chunk
    
    # Set up the mock to return our streaming response
    mock_completion.return_value = MockStreamingResponse()
    
    # Make a request to chat endpoint with streaming
    request_data = {
        "model": "llama3",
        "messages": [
            {"role": "user", "content": "How are you?"}
        ],
        "stream": True
    }
    
    # Use the test client to make a streaming request
    response = test_client.post("/api/chat", json=request_data)
    
    # Check that the response is a streaming response
    assert response.status_code == 200
    assert response.headers["content-type"] == "text/event-stream"
    
    # Parse the streaming response
    chunks = list(response.iter_lines())
    assert len(chunks) >= 3  # We should have at least 3 chunks plus a final done message
    
    # Verify the content of each chunk
    all_content = ""
    for i, chunk in enumerate(chunks):
        if not chunk:  # Skip empty lines
            continue
        
        data = json.loads(chunk)
        if "done" in data and data["done"] is True:
            # This is the final chunk
            continue
        
        assert "model" in data
        assert "response" in data
        assert "done" in data
        assert data["done"] is False
        all_content += data["response"]
    
    # Check that we received the complete message
    assert "I'm doing well!" in all_content
