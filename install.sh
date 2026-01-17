#!/bin/bash
# Install all dependencies for DINOv3 MLP training notebook

echo "Installing dependencies for DINOv3 training..."

# Core ML libraries
pip install torch torchvision torchaudio

# Data science essentials
pip install numpy pandas pillow

# Training utilities
pip install tqdm scikit-learn

# Visualization
pip install matplotlib seaborn

# Jupyter
pip install jupyter notebook ipywidgets

# Optional: If you want to use HuggingFace instead of torch.hub
pip install transformers

echo ""
echo "Done! Make sure you have:"
echo "  1. Python 3.10+ (required for DINOv3)"
echo "  2. DINOv3 repo at: /Users/huntercameronkuperman/Downloads/dinov3-main"
echo ""
echo "Run the notebook cells IN ORDER from top to bottom."
