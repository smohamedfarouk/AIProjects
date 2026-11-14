import subprocess
import time
import sys
from pathlib import Path

from app.common.customexceptions import CustomException
from dotenv import load_dotenv
from app.config.settings import settings
from app.common.logger import get_logger

logger = get_logger()

load_dotenv()


def run_server():
    try:    
        logger.info("Starting backend service...")
        command = [sys.executable, "-m", "uvicorn", "app.backend.api:app", "--host", settings.API_HOST, "--port", settings.API_PORT]
        process = subprocess.Popen(command)
        return process
    except CustomException as e:
        logger.error(f"Problem running backend service server: {e}")
        raise CustomException(f"Problem running backend service server: {e}")

def run_frontend():
    try:
        logger.info("Starting frontend service...")
        frontend_path = Path(__file__).resolve().parent / "frontend" / "ui.py"
        command = [sys.executable, "-m", "streamlit", "run", str(frontend_path)]
        process = subprocess.run(command)
        return process
    except CustomException as e:
        logger.error(f"Problem running frontend service: {e}")
        raise CustomException(f"Problem running frontend service: {e}") 

if __name__ == "__main__":
    server = run_server()
    try:
        time.sleep(2)  # Give the server some time to start before running the frontend
        if server.poll() is not None:
            raise RuntimeError("Backend service failed to start")
        run_frontend()
    finally:
        server.terminate()
        server.wait()

    
    