import uvicorn
import os
from dotenv import load_dotenv
from app.config import DEFAULT_PORT, DEFAULT_HOST, DEFAULT_LOG_LEVEL

# Load environment variables
load_dotenv()

if __name__ == "__main__":
    port = int(os.getenv("PORT", str(DEFAULT_PORT)))
    host = os.getenv("HOST", DEFAULT_HOST)
    log_level = os.getenv("LOG_LEVEL", DEFAULT_LOG_LEVEL).lower()
    
    print(f"Starting Ollama API compatibility server on {host}:{port}")
    
    uvicorn.run(
        "app.main:app",
        host=host,
        port=port,
        reload=True,
        log_level=log_level
    )
