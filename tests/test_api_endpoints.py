"""
Test the main API endpoints of the Ollama API proxy.
"""
import json
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
    """Test the show model endpoint."""
    model_name = "llama3"
    response = test_client.get(f"/api/show?model={model_name}")
    assert response.status_code == 200
    assert "license" in response.json()
    assert "modelfile" in response.json()
    assert "details" in response.json()
    assert model_name in response.json()["modelfile"]


def test_status_endpoint(test_client):
    """Test the status endpoint."""
    response = test_client.get("/api/status")
    assert response.status_code == 200
    assert "status" in response.json()
    assert "running" in response.json()
    assert response.json()["status"] == "ok"


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


import json

def test_unsupported_endpoint_delete(test_client):
    """Test unsupported DELETE endpoint."""
    response = test_client.delete(
        "/api/delete",
        data=json.dumps({"model": "test"}),
        headers={"Content-Type": "application/json"}
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
def test_embeddings_endpoint(mock_embedding, test_client):
    """Test the embeddings endpoint."""
    # Mock litellm embedding response
    mock_response = MagicMock()
    mock_response.data = [MagicMock()]
    mock_response.data[0].embedding = [0.1, 0.2, 0.3, 0.4, 0.5]
    mock_embedding.return_value = mock_response

    # Make a request to embeddings endpoint
    request_data = {
        "model": "mxbai-embed-large",
        "prompt": "Hello, world!"
    }
    response = test_client.post("/api/embeddings", json=request_data)
    
    # Verify response
    assert response.status_code == 200
    assert "embedding" in response.json()
    assert isinstance(response.json()["embedding"], list)
    assert len(response.json()["embedding"]) == 5

    # Verify litellm was called correctly
    mock_embedding.assert_called_once()
    args, kwargs = mock_embedding.call_args
    assert kwargs["model"] == "ollama/mxbai-embed-large"
    assert kwargs["input"] == "Hello, world!"
