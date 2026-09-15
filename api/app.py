# the http layer; it holds no game state between requests

from fastapi import FastAPI

app = FastAPI(title="Whist AI Service")


@app.get("/health")
def health() -> dict:
    # what the hosting platform pings to check the service is alive
    return {"status": "ok"}