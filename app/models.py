"""
Pydantic models for the Ollama API proxy.
"""
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any


class GenerateRequest(BaseModel):
    """Request model for the /api/generate endpoint"""
    model: str
    prompt: str
    images: Optional[List[str]] = None  # List of base64-encoded images
    suffix: Optional[str] = None
    system: Optional[str] = None
    template: Optional[str] = None
    context: Optional[List[int]] = None # Deprecated
    options: Optional[Dict[str, Any]] = Field(default_factory=dict)
    format: Optional[str] = None # Can be "json" or a JSON schema dict
    stream: Optional[bool] = False
    raw: Optional[bool] = False
    keep_alive: Optional[str] = None # E.g., "5m"


class GenerateResponse(BaseModel):
    """Response model for the /api/generate endpoint"""
    model: str
    created_at: str
    response: str # Empty if streamed
    done: bool
    done_reason: Optional[str] = None # E.g., "stop", "length", "load", "error", "unload"
    context: Optional[List[int]] = None # Deprecated, but part of the spec response
    total_duration: Optional[int] = None # Nanoseconds
    load_duration: Optional[int] = None # Nanoseconds
    prompt_eval_count: Optional[int] = None
    prompt_eval_duration: Optional[int] = None # Nanoseconds
    eval_count: Optional[int] = None
    eval_duration: Optional[int] = None # Nanoseconds


class ToolCall(BaseModel):
    """Tool call information"""
    function: Dict[str, Any] # {"name": "...", "arguments": {...}}


class ChatMessage(BaseModel):
    """A chat message"""
    role: str # system, user, assistant, tool
    content: str
    images: Optional[List[str]] = None # List of base64-encoded images
    tool_calls: Optional[List[ToolCall]] = None # Only for assistant messages requesting tool use


class ChatRequest(BaseModel):
    """Request model for the /api/chat endpoint"""
    model: str
    messages: List[ChatMessage]
    tools: Optional[List[Dict[str, Any]]] = None # List of tools
    stream: Optional[bool] = False
    options: Optional[Dict[str, Any]] = Field(default_factory=dict)
    format: Optional[str] = None # Can be "json" or a JSON schema dict
    keep_alive: Optional[str] = None # E.g., "5m"


class ChatResponse(BaseModel):
    """Response model for the /api/chat endpoint"""
    model: str
    created_at: str
    message: ChatMessage # Empty content if streamed
    done: bool
    done_reason: Optional[str] = None # E.g., "stop", "length", "load", "error", "unload", "tool_calls"
    total_duration: Optional[int] = None # Nanoseconds
    load_duration: Optional[int] = None # Nanoseconds
    prompt_eval_count: Optional[int] = None
    prompt_eval_duration: Optional[int] = None # Nanoseconds
    eval_count: Optional[int] = None
    eval_duration: Optional[int] = None # Nanoseconds


class ModelDetails(BaseModel):
    """Model details"""
    format: Optional[str] = None
    family: Optional[str] = None
    parent_model: Optional[str] = None # Added in /api/show response
    format: Optional[str] = None
    family: Optional[str] = None
    families: Optional[List[str]] = None
    parameter_size: Optional[str] = None
    quantization_level: Optional[str] = None


class ModelInfo(BaseModel):
    """Model information"""
    name: str
    modified_at: str
    size: int
    digest: str
    details: ModelDetails


class ListTagsResponse(BaseModel):
    """Response model for the /api/tags endpoint"""
    models: List[ModelInfo]


class EmbeddingRequest(BaseModel):
    """Request model for the /api/embed endpoint"""
    model: str
    input: str | List[str] # Renamed from prompt, can be string or list
    options: Optional[Dict[str, Any]] = Field(default_factory=dict)
    truncate: Optional[bool] = True
    keep_alive: Optional[str] = None


class EmbeddingResponse(BaseModel):
    """Response model for the /api/embed endpoint"""
    model: str # Added model field
    embeddings: List[List[float]] # Changed from embedding: List[float]
    total_duration: Optional[int] = None # Nanoseconds
    load_duration: Optional[int] = None # Nanoseconds
    prompt_eval_count: Optional[int] = None


class ShowModelResponse(BaseModel):
    """Response model for the /api/show endpoint"""
    license: str
    modelfile: str
    parameters: str
    template: str
    details: ModelDetails
    model_info: Optional[Dict[str, Any]] = None # Added model_info field
    capabilities: Optional[List[str]] = None # Added capabilities field


class ShowModelRequest(BaseModel):
    """Request model for the POST /api/show endpoint"""
    model: str
    verbose: Optional[bool] = False


class PsModelInfo(BaseModel):
    """Model information for the /api/ps endpoint"""
    name: str
    model: str
    size: int
    digest: str
    details: ModelDetails
    expires_at: Optional[str] = None # ISO 8601 format
    size_vram: Optional[int] = None


class PsResponse(BaseModel):
    """Response model for the /api/ps endpoint"""
    models: List[PsModelInfo]


# --- Models for Stubbed Endpoints ---

class CreateModelRequest(BaseModel):
    """Request model for POST /api/create"""
    model: str
    path: Optional[str] = None # Path to Modelfile (if used locally, not via API body)
    modelfile: Optional[str] = None # Content of the Modelfile
    stream: Optional[bool] = False
    # Ollama spec has many more fields here (from, files, adapters, etc.)
    # Keeping it simple as it's a stub.


class CopyModelRequest(BaseModel):
    """Request model for POST /api/copy"""
    source: str
    destination: str


class DeleteModelRequest(BaseModel):
    """Request model for DELETE /api/delete"""
    model: str


class PullModelRequest(BaseModel):
    """Request model for POST /api/pull"""
    model: str
    insecure: Optional[bool] = False
    stream: Optional[bool] = False


class PushModelRequest(BaseModel):
    """Request model for POST /api/push"""
    model: str
    insecure: Optional[bool] = False
    stream: Optional[bool] = False


class VersionResponse(BaseModel):
    """Response model for GET /api/version"""
    version: str
