# Ollama API Proxy with LiteLLM

A drop-in replacement for the Ollama API that uses LiteLLM in the backend to route requests to various LLM providers (OpenAI, Anthropic, etc.).

## Features

- Ollama API-compatible endpoints
- Routes requests to various LLM providers using LiteLLM
- Supports streaming responses
- Built with FastAPI for high performance

## Prerequisites

- Python 3.8+
- API keys for the LLM providers you want to use

## Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/ollama-proxy.git
   cd ollama-proxy
   ```

2. Create a virtual environment and install dependencies:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. Create a `.env` file from the template:
   ```bash
   cp .env.example .env
   ```

4. Add your API keys to the `.env` file:
   ```
   OPENAI_API_KEY=your_openai_key
   ANTHROPIC_API_KEY=your_anthropic_key
   # Add other keys as needed
   ```

## Usage

1. Start the server:
   ```bash
   python run.py
   ```

2. The server will be available at `http://localhost:11434` (the same port that Ollama uses by default).

3. Use the API just like you would use the Ollama API:
   ```bash
   curl -X POST http://localhost:11434/api/generate -d '{
     "model": "llama3",
     "prompt": "Tell me a story about a robot learning to paint."
   }'
   ```

## API Endpoints

- `POST /api/generate` - Generate completions
- `POST /api/chat` - Generate chat completions
- `GET /api/models` - List available models
- `GET /api/version` - Get version information

## Configuration

You can configure the server by modifying the `.env` file:

- `PORT` - The port to run the server on (default: 11434)
- `HOST` - The host to bind to (default: 0.0.0.0)
- `LOG_LEVEL` - The logging level (default: INFO)

## Model Mapping

The service maps Ollama model names to the appropriate LiteLLM model names. You can customize this mapping in the `map_to_litellm_model` function in `app/main.py`.

## Development

   aider-google-free --yes-always --auto-test --test-cmd "uv run pytest" --notifications 

## License

MIT

## Acknowledgements

- [Ollama](https://github.com/ollama/ollama) for the API design
- [LiteLLM](https://github.com/BerriAI/litellm) for the unified LLM interface
- [FastAPI](https://fastapi.tiangolo.com/) for the web framework
