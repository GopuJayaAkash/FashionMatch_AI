import torch
import numpy as np
from PIL import Image
from transformers import CLIPModel, CLIPProcessor
from .config import CLIP_MODEL_NAME

class CLIPEncoder:
    def __init__(self, model_name=CLIP_MODEL_NAME):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.model = CLIPModel.from_pretrained(model_name).to(self.device)
        self.model.eval()

    @torch.inference_mode()
    def encode_images(self, images):
        inputs = self.processor(images=images, return_tensors="pt", padding=True).to(self.device)
        features = self.model.get_image_features(**inputs)
        features = features / features.norm(dim=-1, keepdim=True)
        return features.cpu().numpy().astype("float32")

    @torch.inference_mode()
    def encode_texts(self, texts):
        inputs = self.processor(text=texts, return_tensors="pt", padding=True,
                                truncation=True).to(self.device)
        features = self.model.get_text_features(**inputs)
        features = features / features.norm(dim=-1, keepdim=True)
        return features.cpu().numpy().astype("float32")
