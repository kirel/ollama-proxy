"""
Tests for the model mapping functionality.
"""
import sys
import os

# Add the parent directory to the path so we can import the app
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app.main import map_to_litellm_model


def test_exact_model_mapping():
    """Test mapping for models that have an exact match in MODEL_MAPPING."""
    # Test OpenAI models
    assert map_to_litellm_model("gpt-3.5-turbo") == "openai/gpt-3.5-turbo"
    assert map_to_litellm_model("gpt-4") == "openai/gpt-4"
    
    # Test Anthropic models
    assert map_to_litellm_model("claude-3-opus") == "anthropic/claude-3-opus-20240229"
    assert map_to_litellm_model("claude-3-sonnet") == "anthropic/claude-3-sonnet-20240229"
    
    # Test Ollama native models
    assert map_to_litellm_model("llama3") == "ollama/llama3"
    assert map_to_litellm_model("mistral") == "ollama/mistral"


def test_default_model_mapping():
    """Test mapping for models that don't have a specific mapping."""
    # Test with a model that doesn't have a specific mapping
    assert map_to_litellm_model("unknown-model") == "ollama/unknown-model"
    
    # Test with a model name containing special characters
    assert map_to_litellm_model("custom_model-123") == "ollama/custom_model-123"


def test_complex_model_names():
    """Test mapping for more complex model names."""
    # Test with a model name containing a namespace (should be passed directly)
    assert map_to_litellm_model("username/custom-model") == "username/custom-model"
    assert map_to_litellm_model("anthropic/claude-3.5-sonnet-20240620") == "anthropic/claude-3.5-sonnet-20240620"

    # Test with a model name containing multiple colons (should use default mapping as no '/' and no specific map)
    assert map_to_litellm_model("model:tag:subtag") == "ollama/model:tag:subtag"
