import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns 
import cv2

'''Use this class to compute score between two patches'''
class Resnet50Score:
    
    '''Set up environment'''
    def __init__(self):
        
        # Load the pretrained Resnet50 model
        self.model = models.resnet50(pretrained=True)
        
        # Remove the final classification layer
        self.model = nn.Sequential(*list(self.model.children())[:-1]) 
        
        # Set the model to evaluation mode
        self.model.eval()
        
        # Define image transform
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229,0.224, 0.225]),
        ])
        
    """ This function computes embedding of a patch (PIL.Image) """       
    def compute_patch_embedding(self, patch, patch_size=256):
        
        # Check patch sizes
        width, height = patch.size
        if width != patch_size or height != patch_size:
            raise ValueError(f"patch_a and patch_b should be of size {patch_size}")
        
        # load and transform the patch
        patch_t = self.transform(patch).unsqueeze(0)
        
        # Compute feature embeddings
        with torch.no_grad():
            patch_emb = self.model(patch_t)
        
        # convert feature embeddings to 1D NumPy arrays (in cosine similarity, each embedding needed to be 1D vec)
        patch_emb = patch_emb.cpu().numpy().flatten()
        return patch_emb
    
    def compute_similarity_between_embeddings(self, patch_a_emb, patch_b_emb):
        cosine_similarity = np.dot(patch_a_emb, patch_b_emb) / (np.linalg.norm(patch_a_emb) * np.linalg.norm(patch_b_emb))
        return cosine_similarity
    
    """ This function computes the score between two patches.
    Both patches should be a PIL.Image class """
    def compute_similarity_between_patches(self, patch_a, patch_b, patch_size=256):
        patch_a_emb = self.compute_patch_embedding(patch_a, patch_size)
        patch_b_emb = self.compute_patch_embedding(patch_b, patch_size)
        return self.compute_similarity_between_embeddings(patch_a_emb, patch_b_emb)
    
