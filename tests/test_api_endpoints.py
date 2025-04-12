"""
Test the main API endpoints of the Ollama API proxy.
"""
import pytest
from unittest.mock import MagicMock, patch
from datetime import datetime


def test_root_endpoint(test_client):
    """Test the root endpoint."""
    response = test_client.get("/")
    assert response.status_code == 200
    assert "message" in response.json()
    assert "Ollama API compatibility layer" in response.json()["message"]


def test_version_endpoint(test_client):
    """Test the version endpoint."""
    response = test_client.get("/api/version")
    assert response.status_code == 200
    assert "version" in response.json()


def test_models_endpoint(test_client):
    """Test the models endpoint."""
    response = test_client.get("/api/models")
    assert response.status_code == 200
    assert "models" in response.json()
    assert len(response.json()["models"]) > 0
    assert "name" in response.json()["models"][0]
    assert "modified_at" in response.json()["models"][0]
    assert "details" in response.json()["models"][0]


def test_show_model_endpoint(test_client):
    """Test the show model endpoint (POST)."""
    model_name = "llama3"
    request_data = {"model": model_name}
    response = test_client.post("/api/show", json=request_data) # Changed to POST and use JSON body
    assert response.status_code == 200
    assert "license" in response.json()
    assert "modelfile" in response.json()
    assert "details" in response.json()
    assert model_name in response.json()["modelfile"] # Check if original name is in modelfile placeholder


def test_ps_endpoint(test_client): # Renamed test function
    """Test the ps endpoint (formerly status)."""
    response = test_client.get("/api/ps") # Updated path
    assert response.status_code == 200
    assert "models" in response.json() # Updated assertion for PsResponse
    assert isinstance(response.json()["models"], list) # Updated assertion


@pytest.mark.parametrize("endpoint", [
    "/api/create",
    "/api/copy",
    "/api/pull",
    "/api/push"
])
def test_unsupported_endpoints_post(test_client, endpoint):
    """Test unsupported POST endpoints."""
    response = test_client.post(endpoint, json={"model": "test"})
    assert response.status_code == 501
    assert "detail" in response.json()



def test_unsupported_endpoint_delete(test_client):
    """Test unsupported DELETE endpoint."""
    response = test_client.request(
        "DELETE",
        "/api/delete",
        json={"model": "test"}
    )
    assert response.status_code == 501
    assert "detail" in response.json()


@patch('app.main.litellm.completion')
def test_generate_endpoint(mock_completion, test_client):
    """Test the generate endpoint."""
    # Mock litellm completion response
    mock_response = MagicMock()
    mock_response.created = datetime.now().timestamp()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = "This is a test response."
    mock_completion.return_value = mock_response

    # Make a request to generate endpoint
    request_data = {
        "model": "llama3",
        "prompt": "Tell me about Python.",
        "system": "You are a helpful assistant.",
        "stream": False
    }
    response = test_client.post("/api/generate", json=request_data)
    
    # Verify response
    assert response.status_code == 200
    assert "model" in response.json()
    assert "response" in response.json()
    assert "done" in response.json()
    assert response.json()["model"] == "llama3"
    assert response.json()["response"] == "This is a test response."
    assert response.json()["done"] is True

    # Verify litellm was called correctly
    mock_completion.assert_called_once()
    args, kwargs = mock_completion.call_args
    assert kwargs["model"] == "ollama/llama3"
    assert len(kwargs["messages"]) == 2
    assert kwargs["messages"][0]["role"] == "system"
    assert kwargs["messages"][1]["role"] == "user"
    assert kwargs["messages"][1]["content"] == "Tell me about Python."


@patch('app.main.litellm.completion')
def test_chat_endpoint(mock_completion, test_client):
    """Test the chat endpoint."""
    # Mock litellm completion response
    mock_response = MagicMock()
    mock_response.created = datetime.now().timestamp()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = "I'm doing well, how are you?"
    mock_completion.return_value = mock_response

    # Make a request to chat endpoint
    request_data = {
        "model": "llama3",
        "messages": [
            {"role": "user", "content": "Hello, how are you?"}
        ],
        "stream": False
    }
    response = test_client.post("/api/chat", json=request_data)
    
    # Verify response
    assert response.status_code == 200
    assert "model" in response.json()
    assert "message" in response.json()
    assert "done" in response.json()
    assert response.json()["model"] == "llama3"
    assert response.json()["message"]["role"] == "assistant"
    assert response.json()["message"]["content"] == "I'm doing well, how are you?"
    assert response.json()["done"] is True

    # Verify litellm was called correctly
    mock_completion.assert_called_once()
    args, kwargs = mock_completion.call_args
    assert kwargs["model"] == "ollama/llama3"
    assert len(kwargs["messages"]) == 1
    assert kwargs["messages"][0]["role"] == "user"
    assert kwargs["messages"][0]["content"] == "Hello, how are you?"


@patch('app.main.litellm.embedding')
def test_embed_endpoint(mock_embedding, test_client): # Renamed test function
    """Test the embed endpoint (formerly embeddings)."""
    # Mock litellm embedding response
    mock_response = MagicMock()
    mock_response.model = "ollama/mxbai-embed-large" # LiteLLM response includes model
    mock_response.data = [MagicMock()]
    mock_response.data[0].embedding = [0.1, 0.2, 0.3, 0.4, 0.5]
    # Add mock usage data if needed by the response model (total_duration, etc.)
    mock_response.usage = MagicMock()
    mock_response.usage.prompt_tokens = 5 # Example value
    mock_embedding.return_value = mock_response

    # Make a request to embed endpoint
    request_data = {
        "model": "mxbai-embed-large",
        "input": "Hello, world!" # Changed from prompt to input
    }
    response = test_client.post("/api/embed", json=request_data) # Updated path

    # Verify response
    assert response.status_code == 200
    assert "model" in response.json() # Check for model field
    assert "embeddings" in response.json() # Changed from embedding to embeddings
    assert isinstance(response.json()["embeddings"], list)
    assert isinstance(response.json()["embeddings"][0], list) # It's a list of lists
    assert len(response.json()["embeddings"][0]) == 5
    assert response.json()["model"] == "mxbai-embed-large" # Check model name in response

    # Verify litellm was called correctly
    mock_embedding.assert_called_once()
    args, kwargs = mock_embedding.call_args
    assert kwargs["model"] == "ollama/mxbai-embed-large"
    # LiteLLM embedding expects a list, even for single input
    assert kwargs["input"] == ["Hello, world!"]
