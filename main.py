import uvicorn
import os
import sys

# Ensure root directory is in sys.path
sys.path.insert(0, os.path.dirname(__file__))

if __name__ == "__main__":
    print("[Starting] Ultimate-RAG-System ASGI Server on http://localhost:8000 ...")
    uvicorn.run("ui.server:app", host="0.0.0.0", port=8000, reload=True)

