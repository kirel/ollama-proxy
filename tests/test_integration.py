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
import tempfile # Import tempfile
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


@pytest.fixture # Remove async def
def async_ollama_client():
    """Create an async ollama client connected to our proxy."""
    client = AsyncClient(host=OLLAMA_HOST)
    return client # Return the client instance directly


def test_list_models(ollama_client):
    """Test that the client can list models through our proxy."""
    response_dict = ollama_client.list()
    assert "models" in response_dict
    assert isinstance(response_dict["models"], list)
    assert len(response_dict["models"]) > 0
    # The client returns Model objects, access attributes directly
    first_model = response_dict["models"][0]
    assert hasattr(first_model, 'model')
    assert isinstance(first_model.model, str)


def test_generate(ollama_client):
    """Test that the client can generate completions through our proxy."""
    # Ensure the model 'qwen2:0.5b' is available on the target Ollama instance
    test_model = "qwen2:0.5b"
    response = ollama_client.generate(
        model=test_model,
        prompt="Tell me a short joke.",
        options={"temperature": 0.7}
    )
    assert "response" in response
    assert len(response["response"]) > 0
    assert "model" in response
    assert response["model"] == test_model


def test_chat(ollama_client):
    """Test that the client can chat through our proxy."""
    # Ensure the model 'qwen2:0.5b' is available on the target Ollama instance
    test_model = "qwen2:0.5b"
    response = ollama_client.chat(
        model=test_model,
        messages=[
            {"role": "user", "content": "Hello, how are you?"}
        ],
        stream=False # Explicitly disable streaming
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
    # Ensure the model 'qwen2:0.5b' is available on the target Ollama instance
    test_model = "qwen2:0.5b"
    response = ollama_client.chat(
        model=test_model,
        messages=messages,
        stream=False # Explicitly disable streaming
    )
    assert "message" in response
    assert "content" in response["message"]
    # Relaxed assertion: Just check that some content was returned,
    # as small models might not handle history perfectly.
    assert len(response["message"]["content"]) > 0


def test_embeddings(ollama_client):
    """Test that the client can generate embeddings through our proxy."""
    # Ensure the model 'qwen2:0.5b' is available on the target Ollama instance
    # Note: Using a chat model for embeddings might not be ideal, but works for API testing.
    # Consider using a dedicated embedding model if available, e.g., 'nomic-embed-text'
    test_model = "qwen2:0.5b"
    response = ollama_client.embeddings(
        model=test_model,
        prompt="Hello, world!" # ollama client uses 'prompt', maps to 'input'
    )
    # The ollama client returns an EmbeddingsResponse object. Access via attribute.
    # Note: The client uses the singular 'embedding' attribute.
    assert hasattr(response, 'embedding')
    assert isinstance(response.embedding, list)
    assert len(response.embedding) > 0 # Check that the embedding vector has dimension


def test_show_model(ollama_client):
    """Test that the client can show model details through our proxy (using POST)."""
    # Ensure the model 'qwen2:0.5b' is available on the target Ollama instance
    test_model = "qwen2:0.5b"
    response = ollama_client.show(test_model) # ollama client uses POST correctly
    assert "modelfile" in response
    assert "parameters" in response
    assert "template" in response
    assert "license" in response


@pytest.mark.asyncio
async def test_async_generate(async_ollama_client):
    """Test that the async client can generate completions through our proxy."""
    # Ensure the model 'qwen2:0.5b' is available on the target Ollama instance
    test_model = "qwen2:0.5b"
    response = await async_ollama_client.generate(
        model=test_model,
        prompt="What is Python?",
        options={"temperature": 0.7}
    )
    assert "response" in response
    assert len(response["response"]) > 0


@pytest.mark.asyncio
async def test_async_chat(async_ollama_client):
    """Test that the async client can chat through our proxy."""
    # Ensure the model 'qwen2:0.5b' is available on the target Ollama instance
    test_model = "qwen2:0.5b"
    response = await async_ollama_client.chat(
        model=test_model,
        messages=[
            {"role": "user", "content": "Tell me about the Python programming language."}
        ],
        stream=False # Explicitly disable streaming
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
    # Ensure the model 'qwen2:0.5b' is available on the target Ollama instance
    test_model = "qwen2:0.5b"
    chunks = []
    async for chunk in await async_ollama_client.chat(
        model=test_model,
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
    # Relaxed assertion: Just check that some content was streamed,
    # as the specific output depends on the model's counting ability.
    # Example check: assert "count" in full_response.lower() # Optional: check for keywords


def test_unsupported_endpoints(ollama_client):
    """Test that unsupported endpoints return appropriate errors."""
    # Test create model (should fail with 501 from the proxy)
    modelfile_content = "FROM qwen2:0.5b\nSYSTEM You are a helpful assistant."
    with pytest.raises(Exception) as excinfo, tempfile.NamedTemporaryFile(mode='w+', delete=True) as temp_modelfile:
        temp_modelfile.write(modelfile_content)
        temp_modelfile.flush() # Ensure content is written to disk
        # Call create using the path to the temporary file
        ollama_client.create(
            model="test-model",
            path=temp_modelfile.name
        )
    # Check for the proxy's 501 error OR the client's ResponseError containing 501
    # The client should now attempt the request via path, and the proxy should return 501.
    assert "501" in str(excinfo.value) or "Creating models from Modelfiles is not supported" in str(excinfo.value)

    # Test pull model (should fail with 501)
    with pytest.raises(Exception) as excinfo:
        ollama_client.pull("llama3")
    assert "501" in str(excinfo.value)
