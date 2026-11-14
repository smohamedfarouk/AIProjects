from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from app.common.customexceptions import CustomException
from app.common.logger import get_logger
from app.config.settings import settings
from app.core.ai_agent import get_ai_message

logger = get_logger()
app = FastAPI(title=settings.APP_NAME or "LLMOps", version=settings.APP_VERSION or "0.1.0")


class AIMessageRequest(BaseModel):
    model_name: str
    messages: list[str] = Field(min_length=1)
    allow_search: bool = False
    system_prompt: str = ""


@app.post("/chat")
def ai_message_post(request: AIMessageRequest):
    if request.model_name not in settings.ALLOWED_MODELS_NAMES:
        raise HTTPException(status_code=400, detail="Invalid model_name")

    try:
        return get_ai_message(
            request.model_name, request.messages, request.allow_search, request.system_prompt
        )
    except CustomException as error:
        logger.exception("AI message request failed")
        raise HTTPException(status_code=502, detail=str(error)) from error

