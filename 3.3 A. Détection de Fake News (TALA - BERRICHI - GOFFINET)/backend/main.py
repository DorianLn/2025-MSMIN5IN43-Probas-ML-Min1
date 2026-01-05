from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.routes import predict
from app.config import settings

app = FastAPI(title=settings.PROJECT_NAME)

# Configuration CORS pour autoriser React (important !)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # En développement, autorise tout
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Inclusion des routes
app.include_router(predict.router)

@app.get("/")
async def health_check():
    return {"status": "online", "project": settings.PROJECT_NAME}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)