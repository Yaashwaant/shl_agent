"""
Application entry point.
Run with: python main.py
Or: uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
"""
import uvicorn

if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=False,         # Disable reload in prod
        log_level="info",
        access_log=True,
        timeout_keep_alive=30,
    )
