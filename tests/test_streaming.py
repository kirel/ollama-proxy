"""
Tests for the streaming functionality.
"""
import json
from unittest.mock import MagicMock, patch


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
    assert response.headers["content-type"] == "text/event-stream; charset=utf-8"
    
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
    assert response.headers["content-type"] == "text/event-stream; charset=utf-8"
    
    # Parse the streaming response
    chunks = list(response.iter_lines())
    assert len(chunks) >= 4  # 3 content chunks + 1 final done message

    # Verify the content of each chunk
    all_content = ""
    final_chunk_data = None
    intermediate_chunks = []
    for chunk_str in chunks:
        if not chunk_str: continue # Skip potential empty lines

        data = json.loads(chunk_str)
        if data.get("done") is True:
            final_chunk_data = data
        else:
            intermediate_chunks.append(data)
            assert "model" in data
            assert "created_at" in data
            assert "message" in data # Chat uses 'message' key
            assert "role" in data["message"]
            assert "content" in data["message"]
            assert data["message"]["role"] == "assistant"
            assert "done" in data and data["done"] is False
            all_content += data["message"]["content"]

    # Check intermediate chunks
    assert len(intermediate_chunks) == 3
    assert intermediate_chunks[0]["message"]["content"] == "I'm "
    assert intermediate_chunks[1]["message"]["content"] == "doing "
    assert intermediate_chunks[2]["message"]["content"] == "well!"

    # Check that we received the complete message
    assert all_content == "I'm doing well!"

    # Check the final chunk structure (as defined in stream_chat_generator)
    assert final_chunk_data is not None
    assert final_chunk_data["done"] is True
    assert "model" in final_chunk_data
    assert "created_at" in final_chunk_data
    assert "message" in final_chunk_data
    assert "role" in final_chunk_data["message"]
    assert "content" in final_chunk_data["message"] and final_chunk_data["message"]["content"] == ""
    assert "done_reason" in final_chunk_data # Check for placeholder fields
    assert "total_duration" in final_chunk_data
