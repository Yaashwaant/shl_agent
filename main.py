"""
Application entry point.
Run with: python main.py
"""
import uvicorn
import os
from app.core.config import get_settings

settings = get_settings()

if __name__ == "__main__":
    # Render provides PORT environment variable
    port = int(os.environ.get("PORT", settings.port))
    
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=port,
        reload=False,         # Disable reload in prod
        log_level="info",
        access_log=True,
        timeout_keep_alive=30,
    )
