from fastapi import FastAPI
from App.api.routes import router  # Update to lowercase app

app = FastAPI(title="AI Course Generator")
app.include_router(router)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)