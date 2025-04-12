"""
Integration tests for the Ollama API proxy.

These tests require both the Ollama API proxy and a real ollama server to be running.
The tests use the ollama Python client to connect to our proxy and verify that the
responses are compatible with what the ollama client expects.

To run these tests:
1. Start the Ollama API proxy: python run.py
2. Have the API keys for LiteLLM configured
3. Run: pytest tests/test_integration.py -v
"""
import pytest
import os
from ollama import Client, AsyncClient

# Set this to the address of your Ollama API proxy
OLLAMA_HOST = os.environ.get("OLLAMA_PROXY_HOST", "http://localhost:11434")

# Skip the entire module if the integration tests are not enabled
pytestmark = pytest.mark.skipif(
    os.environ.get("RUN_INTEGRATION_TESTS") != "1",
    reason="Integration tests are disabled. Set RUN_INTEGRATION_TESTS=1 to enable."
)


@pytest.fixture
def ollama_client():
    """Create an ollama client connected to our proxy."""
    client = Client(host=OLLAMA_HOST)
    return client


@pytest.fixture
async def async_ollama_client():
    """Create an async ollama client connected to our proxy."""
    client = AsyncClient(host=OLLAMA_HOST)
    return client


def test_list_models(ollama_client):
    """Test that the client can list models through our proxy."""
    models = ollama_client.list()
    assert "models" in models
    assert len(models["models"]) > 0
    assert "name" in models["models"][0]


def test_generate(ollama_client):
    """Test that the client can generate completions through our proxy."""
    response = ollama_client.generate(
        model="llama3",
        prompt="Tell me a short joke.",
        options={"temperature": 0.7}
    )
    assert "response" in response
    assert len(response["response"]) > 0
    assert "model" in response
    assert response["model"] == "llama3"


def test_chat(ollama_client):
    """Test that the client can chat through our proxy."""
    response = ollama_client.chat(
        model="llama3",
        messages=[
            {"role": "user", "content": "Hello, how are you?"}
        ]
    )
    assert "message" in response
    assert "content" in response["message"]
    assert "role" in response["message"]
    assert response["message"]["role"] == "assistant"
    assert len(response["message"]["content"]) > 0


def test_chat_with_history(ollama_client):
    """Test that the client can chat with history through our proxy."""
    messages = [
        {"role": "user", "content": "Hello, my name is John."},
        {"role": "assistant", "content": "Hello John! How can I help you today?"},
        {"role": "user", "content": "What's my name?"}
    ]
    response = ollama_client.chat(
        model="llama3",
        messages=messages
    )
    assert "message" in response
    assert "content" in response["message"]
    # The response should mention "John" since the model should remember the name
    assert "John" in response["message"]["content"]


def test_embeddings(ollama_client):
    """Test that the client can generate embeddings through our proxy."""
    response = ollama_client.embeddings(
        model="mxbai-embed-large",
        prompt="Hello, world!"
    )
    assert "embedding" in response
    assert isinstance(response["embedding"], list)
    assert len(response["embedding"]) > 0


def test_show_model(ollama_client):
    """Test that the client can show model details through our proxy."""
    response = ollama_client.show("llama3")
    assert "modelfile" in response
    assert "parameters" in response
    assert "template" in response
    assert "license" in response


@pytest.mark.asyncio
async def test_async_generate(async_ollama_client):
    """Test that the async client can generate completions through our proxy."""
    response = await async_ollama_client.generate(
        model="llama3",
        prompt="What is Python?",
        options={"temperature": 0.7}
    )
    assert "response" in response
    assert len(response["response"]) > 0


@pytest.mark.asyncio
async def test_async_chat(async_ollama_client):
    """Test that the async client can chat through our proxy."""
    response = await async_ollama_client.chat(
        model="llama3",
        messages=[
            {"role": "user", "content": "Tell me about the Python programming language."}
        ]
    )
    assert "message" in response
    assert "content" in response["message"]
    assert len(response["message"]["content"]) > 0


@pytest.mark.asyncio
async def test_streaming_chat(async_ollama_client):
    """Test that the async client can stream chat responses through our proxy."""
    messages = [
        {"role": "user", "content": "Count from 1 to 5."}
    ]
    
    chunks = []
    async for chunk in await async_ollama_client.chat(
        model="llama3",
        messages=messages,
        stream=True
    ):
        chunks.append(chunk)
        assert "message" in chunk
        assert "content" in chunk["message"]
    
    assert len(chunks) > 1  # Make sure we got multiple chunks
    # Combine all chunks to get the full response
    full_response = "".join([chunk["message"]["content"] for chunk in chunks])
    assert len(full_response) > 0
    # The response should contain numbers 1 through 5
    for num in range(1, 6):
        assert str(num) in full_response


def test_unsupported_endpoints(ollama_client):
    """Test that unsupported endpoints return appropriate errors."""
    # Test create model (should fail with 501)
    with pytest.raises(Exception) as excinfo:
        ollama_client.create(
            model="test-model",
            modelfile="FROM llama3\nSYSTEM You are a helpful assistant."
        )
    assert "501" in str(excinfo.value)
    
    # Test pull model (should fail with 501)
    with pytest.raises(Exception) as excinfo:
        ollama_client.pull("llama3")
    assert "501" in str(excinfo.value)
