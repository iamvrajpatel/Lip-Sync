import logging

from app_factory import create_app


logging.basicConfig(level=logging.INFO)
app = create_app()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app=app, host="0.0.0.0", port=8000, log_level="info")
