"""
DINOv3 Table Classifier - Inference Script

Usage:
    from inference import TableClassifier

    classifier = TableClassifier("weights/model.pt")
    result = classifier.predict("path/to/image.jpg")
    # {'class': 'dirty', 'confidence': 0.87, 'probabilities': {'clean': 0.05, 'dirty': 0.87, 'occupied': 0.08}}
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from transformers import AutoModel
from typing import Union
from pathlib import Path


# HuggingFace model IDs
HF_MODELS = {
    "dinov3_vits16": "facebook/dinov3-vits16-pretrain-lvd1689m",
    "dinov3_vitb16": "facebook/dinov3-vitb16-pretrain-lvd1689m",
    "dinov3_vitl16": "facebook/dinov3-vitl16-pretrain-lvd1689m",
}

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class AttentionPool(nn.Module):
    def __init__(self, dim, hidden=128):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 1)
        )

    def forward(self, x):
        weights = F.softmax(self.attn(x), dim=1)
        return (weights * x).sum(dim=1)


class DINOv3Classifier(nn.Module):
    def __init__(self, backbone, embed_dim, num_classes, dropout=0.4, use_attn_pool=True):
        super().__init__()
        self.backbone = backbone
        self.backbone.eval()
        for p in backbone.parameters():
            p.requires_grad = False

        self.use_attn_pool = use_attn_pool
        if use_attn_pool:
            self.attn_pool = AttentionPool(embed_dim)

        feat_dim = embed_dim * 2
        self.head = nn.Sequential(
            nn.LayerNorm(feat_dim),
            nn.Linear(feat_dim, 512),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(512, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes)
        )
        self.embed_dim = embed_dim

    def forward(self, x):
        with torch.no_grad():
            out = self.backbone(x)
            if hasattr(out, 'last_hidden_state'):
                cls_token = out.last_hidden_state[:, 0]
                patches = out.last_hidden_state[:, 1:]
            else:
                cls_token = out['x_norm_clstoken']
                patches = out['x_norm_patchtokens']

        if self.use_attn_pool:
            pooled = self.attn_pool(patches)
        else:
            pooled = patches.mean(dim=1)

        features = torch.cat([cls_token, pooled], dim=1)
        return self.head(features)

    def eval(self):
        super().eval()
        self.backbone.eval()
        return self


class TableClassifier:
    """
    Table state classifier using DINOv3 backbone.

    Args:
        weights_path: Path to the .pt checkpoint file
        device: 'cuda', 'mps', 'cpu', or None (auto-detect)
    """

    def __init__(self, weights_path: str, device: str = None):
        # Auto-detect device
        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            elif torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        self.device = torch.device(device)

        # Load checkpoint
        checkpoint = torch.load(weights_path, map_location=self.device)

        self.id2label = checkpoint['id2label']
        self.label2id = {v: k for k, v in self.id2label.items()}
        embed_dim = checkpoint['embed_dim']
        backbone_name = checkpoint['backbone']
        use_attn_pool = checkpoint.get('use_attn_pool', True)

        # Load backbone from HuggingFace
        hf_model_id = HF_MODELS.get(backbone_name, backbone_name)
        print(f"Loading backbone: {hf_model_id}")
        backbone = AutoModel.from_pretrained(hf_model_id, trust_remote_code=True)

        # Build model
        num_classes = len(self.id2label)
        self.model = DINOv3Classifier(
            backbone=backbone,
            embed_dim=embed_dim,
            num_classes=num_classes,
            use_attn_pool=use_attn_pool
        )

        # Load weights
        self.model.head.load_state_dict(checkpoint['head'])
        if use_attn_pool and 'attn_pool' in checkpoint:
            self.model.attn_pool.load_state_dict(checkpoint['attn_pool'])

        self.model.to(self.device)
        self.model.eval()

        # Preprocessing
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
        ])

        print(f"Model loaded on {self.device}")
        print(f"Classes: {list(self.id2label.values())}")

    @torch.no_grad()
    def predict(self, image: Union[str, Path, Image.Image]) -> dict:
        """
        Predict table state from image.

        Args:
            image: File path or PIL Image

        Returns:
            {
                'class': 'clean' | 'dirty' | 'occupied',
                'confidence': float (0-1),
                'probabilities': {'clean': float, 'dirty': float, 'occupied': float}
            }
        """
        # Load image if path
        if isinstance(image, (str, Path)):
            image = Image.open(image)

        if image.mode != "RGB":
            image = image.convert("RGB")

        # Preprocess and predict
        x = self.transform(image).unsqueeze(0).to(self.device)
        logits = self.model(x)
        probs = F.softmax(logits, dim=1)[0]

        # Get prediction
        pred_idx = probs.argmax().item()
        pred_class = self.id2label[pred_idx]
        confidence = probs[pred_idx].item()

        return {
            'class': pred_class,
            'confidence': round(confidence, 4),
            'probabilities': {
                self.id2label[i]: round(p.item(), 4)
                for i, p in enumerate(probs)
            }
        }

    @torch.no_grad()
    def predict_batch(self, images: list) -> list:
        """Predict on multiple images."""
        return [self.predict(img) for img in images]


# CLI usage
if __name__ == "__main__":
    import sys

    if len(sys.argv) < 3:
        print("Usage: python inference.py <weights.pt> <image.jpg>")
        sys.exit(1)

    weights_path = sys.argv[1]
    image_path = sys.argv[2]

    classifier = TableClassifier(weights_path)
    result = classifier.predict(image_path)

    print(f"\nPrediction: {result['class']} ({result['confidence']:.1%})")
    print(f"Probabilities: {result['probabilities']}")
