from fastapi import FastAPI, HTTPException, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
import litellm
import json
import logging
from typing import List, Dict
from datetime import datetime # Import datetime
import uvicorn
from dotenv import load_dotenv
from app.config import MODEL_MAPPING, LITELLM_CONFIG
from app.models import (
    GenerateRequest, GenerateResponse, ChatMessage, ChatRequest, ChatResponse,
    ModelInfo, ListTagsResponse, ModelDetails, EmbeddingRequest, EmbeddingResponse, # Added ListTagsResponse
    ShowModelResponse, ShowModelRequest, PsResponse, CreateModelRequest, # Added VersionResponse and CreateModelRequest
    CopyModelRequest, DeleteModelRequest, PullModelRequest, PushModelRequest # Added stub request models
)

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="Ollama API Compatibility Layer",
    description="A drop-in replacement for Ollama API using LiteLLM",
    version="0.1.0",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configure LiteLLM
def configure_litellm():
    # You can add your API keys directly or via environment variables
    # LiteLLM will pick up environment variables like OPENAI_API_KEY, ANTHROPIC_API_KEY, etc.
    
    # Set verbosity level
    litellm.verbose = LITELLM_CONFIG.get("verbose", False)
    
    # Apply any other configuration from LITELLM_CONFIG
    for key, value in LITELLM_CONFIG.items():
        if key != "verbose":  # Already handled above
            if hasattr(litellm, f"set_{key}"):
                getattr(litellm, f"set_{key}")(value)
            else:
                logger.warning(f"Unknown LiteLLM configuration option: {key}")

configure_litellm()

# Helper functions defined above

# Helper functions
def map_to_litellm_model(model_name: str) -> str:
    """
    Map Ollama model names to LiteLLM compatible model names.
    If the name contains '/', pass it directly.
    Otherwise, check for convenience mappings or apply the default 'ollama/' prefix.
    """
    # If the model name already contains a '/', assume it's a direct LiteLLM model string
    if "/" in model_name:
        logger.info(f"Using model name directly: {model_name}")
        return model_name

    # If no '/', check for an exact convenience mapping (excluding the default lambda)
    if model_name in MODEL_MAPPING and not callable(MODEL_MAPPING[model_name]):
        mapped_name = MODEL_MAPPING[model_name]
        logger.info(f"Mapped convenience model '{model_name}' to '{mapped_name}'")
        return mapped_name

    # If no '/' and no specific mapping, assume it's a direct Ollama model name.
    # Do NOT apply the 'ollama/' prefix here, let LiteLLM handle it based on its provider detection.
    logger.info(f"No specific mapping found for '{model_name}'. Using it directly for LiteLLM.")
    return model_name

def convert_chat_to_litellm_format(messages: List[ChatMessage]) -> List[Dict[str, str]]:
    """Convert Ollama chat messages to LiteLLM format."""
    return [{"role": msg.role, "content": msg.content} for msg in messages]

async def stream_generate_generator(stream):
    """Generate streaming responses for /api/generate in Ollama format."""
    final_model = None
    final_created_at = None
    async for chunk in stream:
        if hasattr(chunk, "choices") and len(chunk.choices) > 0:
            choice = chunk.choices[0]
            final_model = chunk.model # Store model for final chunk
            final_created_at = datetime.fromtimestamp(chunk.created).isoformat() + "Z" # Store and format timestamp

            if hasattr(choice, "delta") and hasattr(choice.delta, "content"):
                content = choice.delta.content
                if content:
                    # Format intermediate chunk for /api/generate
                    yield json.dumps({
                        "model": final_model,
                        "created_at": final_created_at,
                        "response": content,
                        "done": False
                    }) + "\n"

    # Final message indicating completion for /api/generate
    # Mimics Ollama's final streaming chunk structure
    if final_model and final_created_at:
        yield json.dumps({
            "model": final_model,
            "created_at": final_created_at,
            "response": "", # Empty response in final chunk
            "done": True,
            # Add other fields if available from LiteLLM usage info later
            "context": [], # Placeholder
            "total_duration": 0, # Placeholder
            "load_duration": 0, # Placeholder
            "prompt_eval_count": 0, # Placeholder
            "prompt_eval_duration": 0, # Placeholder
            "eval_count": 0, # Placeholder
            "eval_duration": 0 # Placeholder
        }) + "\n"
    else:
         # Fallback if stream was empty or malformed
         yield json.dumps({"done": True}) + "\n"


async def stream_chat_generator(stream):
    """Generate streaming responses for /api/chat in Ollama format."""
    final_model = None
    final_created_at = None
    async for chunk in stream:
        if hasattr(chunk, "choices") and len(chunk.choices) > 0:
            choice = chunk.choices[0]
            final_model = chunk.model # Store model for final chunk
            final_created_at = datetime.fromtimestamp(chunk.created).isoformat() + "Z" # Store and format timestamp

            if hasattr(choice, "delta") and hasattr(choice.delta, "content"):
                content = choice.delta.content
                if content:
                    # Format intermediate chunk for /api/chat
                    yield json.dumps({
                        "model": final_model,
                        "created_at": final_created_at,
                        "message": {
                            "role": "assistant", # Assuming assistant role for delta content
                            "content": content
                        },
                        "done": False
                    }) + "\n"

    # Final message indicating completion for /api/chat
    # Mimics Ollama's final streaming chunk structure
    if final_model and final_created_at:
        yield json.dumps({
            "model": final_model,
            "created_at": final_created_at,
            "message": { # Empty message content in final chunk
                "role": "assistant",
                "content": ""
            },
            "done": True,
            "done_reason": "stop", # Default reason
            # Add other fields if available from LiteLLM usage info later
            "total_duration": 0, # Placeholder
            "load_duration": 0, # Placeholder
            "prompt_eval_count": 0, # Placeholder
            "prompt_eval_duration": 0, # Placeholder
            "eval_count": 0, # Placeholder
            "eval_duration": 0 # Placeholder
        }) + "\n"
    else:
        # Fallback if stream was empty or malformed
        yield json.dumps({"done": True}) + "\n"


# API Routes
@app.get("/")
async def root_get():
    """Handle GET requests to the root path."""
    return {"message": "Ollama API compatibility layer using LiteLLM"}

@app.head("/")
async def root_head():
    """Handle HEAD requests to the root path for health checks."""
    return Response(status_code=200)

@app.post("/api/generate", response_model=GenerateResponse)
async def generate(request: GenerateRequest):
    """Generate completions (Ollama's /api/generate endpoint)"""
    try:
        model = map_to_litellm_model(request.model)
        logger.info(f"Generating completion with model: {model}")

        # Handle Ollama's model loading behavior when prompt is empty
        if not request.prompt:
            logger.info(f"Empty prompt received for model {request.model}. Treating as load request.")
            # Simulate model load response based on Ollama spec
            created_time = datetime.now().isoformat() + "Z"
            return GenerateResponse(
                model=request.model,
                created_at=created_time,
                response="",
                done=True,
                done_reason="load" # Indicate it was a load operation
            )

        messages = [{"role": "system", "content": request.system}] if request.system else []
        messages.append({"role": "user", "content": request.prompt})

        # Set parameters based on request options
        params = {}
        if request.options:
            # Map Ollama options to LiteLLM parameters (expand as needed)
            if "temperature" in request.options:
                params["temperature"] = request.options["temperature"]
            if "top_p" in request.options:
                params["top_p"] = request.options["top_p"]
            if "max_tokens" in request.options:
                params["max_tokens"] = request.options["max_tokens"]
        
        # Handle streaming
        if request.stream:
            stream = await litellm.acompletion( # Use acompletion
                model=model,
                messages=messages,
                stream=True,
                **params
            )
            return StreamingResponse(
                stream_generate_generator(stream), # Use the generate-specific generator
                media_type="text/event-stream"
            )
        # Non-streaming response
        response = await litellm.acompletion( # Use acompletion
            model=model,
            messages=messages,
            **params
        )
        
        # Get response content
        content = response.choices[0].message.content
        
        # Map response to Ollama format
        created_time = datetime.fromtimestamp(response.created).isoformat() + "Z"
        return GenerateResponse(
            model=request.model,
            created_at=created_time, # Format timestamp
            response=content,
            done=True,
            context=[],  # Ollama returns context here, but we don't have an equivalent
            total_duration=0,  # Would need to measure this
            load_duration=0,
            prompt_eval_count=0,
            prompt_eval_duration=0,
            eval_count=len(content.split()),
            eval_duration=0
        )
        
    except Exception as e:
        logger.error(f"Error in generate: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Chat completions (Ollama's /api/chat endpoint)"""
    try:
        model = map_to_litellm_model(request.model)
        logger.info(f"Generating chat completion with model: {model}")
        logger.info(f"Chat request stream parameter: {request.stream}") # Add logging

        # Handle Ollama's model loading behavior when messages list is empty
        if not request.messages:
            logger.info(f"Empty messages list received for model {request.model}. Treating as load request.")
            # Simulate model load response based on Ollama spec
            created_time = datetime.now().isoformat() + "Z"
            return ChatResponse(
                model=request.model,
                created_at=created_time,
                message=ChatMessage(role="assistant", content=""), # Empty assistant message
                done=True,
                done_reason="load" # Indicate it was a load operation
            )

        messages = convert_chat_to_litellm_format(request.messages)

        # Set parameters based on request options
        params = {}
        if request.options:
            if "temperature" in request.options:
                params["temperature"] = request.options["temperature"]
            if "top_p" in request.options:
                params["top_p"] = request.options["top_p"]
            if "max_tokens" in request.options:
                params["max_tokens"] = request.options["max_tokens"]
        
        # Handle streaming
        if request.stream:
            stream = await litellm.acompletion( # Use acompletion
                model=model,
                messages=messages,
                stream=True,
                **params
            )
            return StreamingResponse(
                stream_chat_generator(stream), # Use the chat-specific generator
                media_type="text/event-stream"
            )
        # Non-streaming response
        response = await litellm.acompletion( # Use acompletion
            model=model,
            messages=messages,
            **params
        )
        
        # Get response content
        content = response.choices[0].message.content
        
        # Map response to Ollama format
        created_time = datetime.fromtimestamp(response.created).isoformat() + "Z"
        return ChatResponse(
            model=request.model,
            created_at=created_time, # Format timestamp
            message=ChatMessage(
                role="assistant",
                content=content
            ),
            done=True,
            total_duration=0,
            load_duration=0,
            prompt_eval_count=0,
            prompt_eval_duration=0,
            eval_count=len(content.split()),
            eval_duration=0
        )
        
    except Exception as e:
        logger.error(f"Error in chat: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/models", response_model=ListTagsResponse) # Renamed from ListModelsResponse
async def list_models():
    """List available models (mimicking Ollama's /api/models endpoint)"""
    # Note: Ollama spec uses /api/tags for this. We might want to align later.
    try:
        # In a real implementation, you'd return available models from LiteLLM
        # For now, we'll return a placeholder response with sample models
        return ListTagsResponse( # Renamed from ListModelsResponse
            models=[
                ModelInfo(
                    name="llama3",
                    modified_at="2023-11-04T14:56:49Z",
                    size=3791730276,
                    digest="sha256:b315144c8e8d286e96e2c38232d1ba5158726c9292419a4371a4531d5b36d2e2",
                    details=ModelDetails(
                        format="gguf",
                        family="llama",
                        families=["llama"],
                        parameter_size="8B",
                        quantization_level="Q5_K_M"
                    )
                ),
                ModelInfo(
                    name="mistral",
                    modified_at="2023-11-03T10:23:15Z",
                    size=4928473102,
                    digest="sha256:a892f82058797e288135da32238bee3a2ff7cbc01d3d31da83091e6bcd8467c7",
                    details=ModelDetails(
                        format="gguf",
                        family="mistral",
                        families=["mistral"],
                        parameter_size="7B",
                        quantization_level="Q4_K_M"
                    )
                ),
                ModelInfo(
                    name="claude-3-haiku",
                    modified_at="2024-03-15T08:45:30Z",
                    size=0,  # Virtual model, no size
                    digest="sha256:virtual",
                    details=ModelDetails(
                        format="api",
                        family="claude",
                        families=["claude"],
                        parameter_size="unknown",
                        quantization_level="none"
                    )
                )
            ]
        )
    except Exception as e:
        logger.error(f"Error listing models: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Additional Ollama endpoints (to be implemented)
@app.get("/api/version")
async def version():
    """Return version information"""
    return {"version": "0.1.0", "build": "ollama-litellm-proxy"}

@app.get("/api/tags")
async def tags():
    """List available model tags"""
    # Placeholder - implement as needed
    return {"models": []}

# Embeddings endpoint (Ollama spec uses /api/embed)
@app.post("/api/embed", response_model=EmbeddingResponse) # Changed path
async def generate_embeddings(request: EmbeddingRequest):
    """Generate embeddings for text (Ollama's /api/embed endpoint)"""
    try:
        # Map the model name to LiteLLM format
        litellm_model = map_to_litellm_model(request.model)
        logger.info(f"Generating embeddings with model: {litellm_model}")

        # Prepare input for LiteLLM (always expects a list)
        input_list = [request.input] if isinstance(request.input, str) else request.input

        # Generate embeddings using LiteLLM
        response = litellm.embedding(
            model=litellm_model,
            input=input_list, # Use input_list
            # Pass other options if needed: request.options, request.truncate, request.keep_alive
        )

        # Extract embeddings from the response
        embeddings_list = [item.embedding for item in response.data]

        # Format response in Ollama format
        return EmbeddingResponse(
            model=request.model, # Return original model name requested
            embeddings=embeddings_list, # Use correct field name
            # Populate duration/count fields if available from response.usage
            prompt_eval_count=response.usage.prompt_tokens if hasattr(response, 'usage') else None,
            # total_duration and load_duration are harder to get accurately here
        )
    except Exception as e:
        logger.error(f"Error generating embeddings: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Show model information (Ollama spec uses POST)
@app.post("/api/show", response_model=ShowModelResponse) # Changed method to POST
async def show_model(request: ShowModelRequest): # Changed signature to use request body
    """Show model information (Ollama's POST /api/show endpoint)"""
    try:
        # Map the model name to LiteLLM format
        litellm_model = map_to_litellm_model(request.model) # Get model from request body
        logger.info(f"Showing information for model: {litellm_model} (verbose={request.verbose})")

        # Get the model family (assuming format is "family:version")
        family = request.model.split(":")[0] if ":" in request.model else request.model

        # Return placeholder model information
        # In a real implementation, we would get this from LiteLLM or other source
        # The 'verbose' parameter should ideally return more detailed info if True
        return ShowModelResponse(
            license="MIT",
            modelfile=f"FROM {request.model}\n", # Use request.model here
            parameters="default parameters",
            template="{{ .Prompt }}",
            details=ModelDetails(
                format="gguf",
                family=family,
                families=[family],
                parameter_size="Unknown",
                quantization_level="Unknown"
            )
        )
    except Exception as e:
        logger.error(f"Error showing model information: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Status endpoint (Ollama spec uses /api/ps)
@app.get("/api/ps", response_model=PsResponse) # Changed path and response model
async def ps(): # Renamed function
    """List running models (Ollama's /api/ps endpoint)"""
    try:
        # In a real implementation, we would get this from LiteLLM or manage state
        # Returning an empty list for now
        return PsResponse(
            models=[]
        )
    except Exception as e:
        logger.error(f"Error getting running models: {str(e)}") # Updated log message
        raise HTTPException(status_code=500, detail=str(e))

# ----- Model Management Endpoints (Stubs) -----

# Create model stub
@app.post("/api/create")
async def create_model(request_data: CreateModelRequest): # Changed signature to use Pydantic model
    """Create a model from a Modelfile"""
    try:
        # This functionality is Ollama-specific and not supported by LiteLLM
        # We don't need to use request_data, just acknowledge it for routing.
        error_message = "Creating models from Modelfiles is not supported in this proxy implementation"
        logger.warning(error_message)
        # Return 501 Not Implemented with informative message
        raise HTTPException(status_code=501, detail=error_message)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in create model endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Copy model stub
@app.post("/api/copy")
async def copy_model(request_data: CopyModelRequest): # Use specific model
    """Copy a model"""
    try:
        # This functionality is Ollama-specific and not supported by LiteLLM
        error_message = "Copying models is not supported in this proxy implementation"
        logger.warning(error_message)
        # Return 501 Not Implemented with informative message
        raise HTTPException(status_code=501, detail=error_message)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in copy model endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Delete model stub
@app.delete("/api/delete")
async def delete_model(request_data: DeleteModelRequest): # Use specific model
    """Delete a model"""
    try:
        # This functionality is Ollama-specific and not supported by LiteLLM
        error_message = "Deleting models is not supported in this proxy implementation"
        logger.warning(error_message)
        # Return 501 Not Implemented with informative message
        raise HTTPException(status_code=501, detail=error_message)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in delete model endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Pull model stub
@app.post("/api/pull")
async def pull_model(request_data: PullModelRequest): # Use specific model
    """Pull a model from a registry"""
    try:
        # This functionality is Ollama-specific and not supported by LiteLLM
        error_message = "Pulling models from a registry is not supported in this proxy implementation"
        logger.warning(error_message)
        # Return 501 Not Implemented with informative message
        raise HTTPException(status_code=501, detail=error_message)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in pull model endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Push model stub
@app.post("/api/push")
async def push_model(request_data: PushModelRequest): # Use specific model
    """Push a model to a registry"""
    try:
        # This functionality is Ollama-specific and not supported by LiteLLM
        error_message = "Pushing models to a registry is not supported in this proxy implementation"
        logger.warning(error_message)
        # Return 501 Not Implemented with informative message
        raise HTTPException(status_code=501, detail=error_message)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in push model endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run("app.main:app", host="0.0.0.0", port=11434, reload=True)
