"""
Configuration for the Ollama API proxy.
"""

# Model mapping from Ollama model names to LiteLLM model names
MODEL_MAPPING = {
    # Default mapping (prefix with 'ollama/')
    "default": lambda model: f"ollama/{model}",
    
    # OpenAI model mappings
    "gpt-3.5-turbo": "openai/gpt-3.5-turbo",
    "gpt-4": "openai/gpt-4",
    "gpt-4-turbo": "openai/gpt-4-turbo-preview",
    
    # Anthropic model mappings
    "claude-3-opus": "anthropic/claude-3-opus-20240229",
    "claude-3-sonnet": "anthropic/claude-3-sonnet-20240229",
    "claude-3-haiku": "anthropic/claude-3-haiku-20240307",
    "claude-3.5": "anthropic/claude-3.5-sonnet-20240620", # Added mapping for 3.5
    "anthropic/claude-3.5": "anthropic/claude-3.5-sonnet-20240620", # Added mapping for namespaced 3.5

    # Google model mappings
    "gemini-pro": "google/gemini-pro",
    "gemini-ultra": "google/gemini-1.5-pro",
    
    # Ollama native models (keep as is but with prefix)
    "llama3": "ollama/llama3",
    "mistral": "ollama/mistral",
    "mixtral": "ollama/mixtral",
    "phi": "ollama/phi",
}

# Server configuration
DEFAULT_PORT = 11434
DEFAULT_HOST = "0.0.0.0"
DEFAULT_LOG_LEVEL = "info"

# LiteLLM configuration
LITELLM_CONFIG = {
    "verbose": False,
    # Add any other LiteLLM configuration options here
}
