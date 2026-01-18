import uvicorn
from fastapi import FastAPI
from request_types.detect_request import detect

class CrowdDetectionAPI(FastAPI):
    def __init__(self) -> None:
        super().__init__()
        self.add_api_route("/detect", detect, methods=["POST"])


def start_api(host, port):
    uvicorn.run(
        app=CrowdDetectionAPI(),
        host=host,
        port=port
    )