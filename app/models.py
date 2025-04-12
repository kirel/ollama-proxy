"""
Pydantic models for the Ollama API proxy.
"""
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any


class GenerateRequest(BaseModel):
    """Request model for the /api/generate endpoint"""
    model: str
    prompt: str
    system: Optional[str] = None
    template: Optional[str] = None
    context: Optional[List[int]] = None
    options: Optional[Dict[str, Any]] = Field(default_factory=dict)
    format: Optional[str] = None
    stream: Optional[bool] = False
    raw: Optional[bool] = False


class GenerateResponse(BaseModel):
    """Response model for the /api/generate endpoint"""
    model: str
    created_at: str
    response: str
    done: bool
    context: Optional[List[int]] = None
    total_duration: Optional[int] = 0
    load_duration: Optional[int] = 0
    prompt_eval_count: Optional[int] = 0
    prompt_eval_duration: Optional[int] = 0
    eval_count: Optional[int] = 0
    eval_duration: Optional[int] = 0


class ChatMessage(BaseModel):
    """A chat message"""
    role: str
    content: str


class ChatRequest(BaseModel):
    """Request model for the /api/chat endpoint"""
    model: str
    messages: List[ChatMessage]
    stream: Optional[bool] = False
    options: Optional[Dict[str, Any]] = Field(default_factory=dict)
    format: Optional[str] = None
    template: Optional[str] = None


class ChatResponse(BaseModel):
    """Response model for the /api/chat endpoint"""
    model: str
    created_at: str
    message: ChatMessage
    done: bool
    total_duration: Optional[int] = 0
    load_duration: Optional[int] = 0
    prompt_eval_count: Optional[int] = 0
    prompt_eval_duration: Optional[int] = 0
    eval_count: Optional[int] = 0
    eval_duration: Optional[int] = 0


class ModelDetails(BaseModel):
    """Model details"""
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


class ListModelsResponse(BaseModel):
    """Response model for the /api/models endpoint"""
    models: List[ModelInfo]


class EmbeddingRequest(BaseModel):
    """Request model for the /api/embeddings endpoint"""
    model: str
    prompt: str
    options: Optional[Dict[str, Any]] = Field(default_factory=dict)


class EmbeddingResponse(BaseModel):
    """Response model for the /api/embeddings endpoint"""
    embedding: List[float]


class ShowModelResponse(BaseModel):
    """Response model for the /api/show endpoint"""
    license: str
    modelfile: str
    parameters: str
    template: str
    details: ModelDetails


class StatusResponse(BaseModel):
    """Response model for the /api/status endpoint"""
    status: str
    running: List[Dict[str, Any]]
    duration: int
