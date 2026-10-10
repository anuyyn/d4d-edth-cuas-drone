import cv2
import numpy as np
import torch
import open_clip
from PIL import Image

PROMPTS = {
    "새": ["a photo of a bird flying in the sky", "a seagull flying in the sky", "a bird with spread wings",
           "a heron flying over trees"],
    "풍선": ["a photo of a balloon floating in the sky", "two large white balloons floating in the sky",
            "a balloon with a bag hanging below it", "a pair of balloons tied together"],
    "무인기": ["a photo of a drone flying in the sky", "a small unmanned aircraft flying",
            "a fixed-wing drone airplane in the air", "a quadcopter drone"],
    "기타": ["an empty blue sky", "a building", "trees and grass", "a cloud in the sky", "a road with cars",
           "a crowd of people", "a roof of a house"],
}


class ClipClassifier:
    def __init__(self, model="ViT-B-32-quickgelu", pretrained="openai", threads=2):
        torch.set_num_threads(threads)
        self.model, _, self.pre = open_clip.create_model_and_transforms(model, pretrained=pretrained)
        self.model.eval()
        tok = open_clip.get_tokenizer(model)
        self.names, texts = [], []
        for cls, ps in PROMPTS.items():
            for p in ps:
                self.names.append(cls)
                texts.append(p)
        with torch.no_grad():
            t = self.model.encode_text(tok(texts))
            self.t = t / t.norm(dim=-1, keepdim=True)

    @torch.no_grad()
    def classify(self, bgr):
        img = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        x = self.pre(img).unsqueeze(0)
        f = self.model.encode_image(x)
        f = f / f.norm(dim=-1, keepdim=True)
        p = (100.0 * f @ self.t.T).softmax(dim=-1)[0].numpy()
        agg = {}
        for n, v in zip(self.names, p):
            agg[n] = agg.get(n, 0.0) + float(v)
        cls = max(agg, key=agg.get)
        return cls, agg[cls], agg
