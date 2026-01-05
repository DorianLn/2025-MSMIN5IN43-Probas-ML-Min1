import torch
import os
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    PROJECT_NAME: str = "FactGuard API"
    # Utilise le GPU si disponible
    DEVICE: int = 0 if torch.cuda.is_available() else -1
    
    # Tes dépôts Hugging Face
    MODELS: dict = {
        "camembert": "LamT45/camenbert_fakenews_model_final",
        "roberta": "LamT45/roberta-fake-news-ENG-Final",
        "bert": "LamT45/ENG_Bert_fake_news_model_final"
    }

settings = Settings()