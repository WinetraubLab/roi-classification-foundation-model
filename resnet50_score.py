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

class Resnet50Score:
    
    def __init__(self):
        self.model = models.resnet50(pretrained=True)
        self.model = nn.Sequential(*list(self.model.children())[:-1]) #  Remove the final classification layer
        self.model.eval()
        
        self.preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229,0.224, 0.225]),
        ])
            
    def extract_embeddings(self, image):
        '''
        Extracts the embedding from the image using the ResNet50 model.
        '''
        # load and preprocess the image
        img_tensor = self.preprocess(image).unsqueeze(0)
        
        # extract embeddings
        with torch.no_grad():
            embeddings = self.model(img_tensor)
        return embeddings
    
    
    def extract_patches(self, image_patch, patch_folder, embedding_folder, patch_size = 256):
        '''
        Extracts image patches from the Test images.
        '''
        os.makedirs(patch_folder, exist_ok=True)
        os.makedirs(embedding_folder, exist_ok=True)
        
        # open the large test image
        image = Image.open(image_patch).convert('RGB')
        
        # calculate the num of patches in each dimension
        width, height = image.size
        num_patches_x = width // patch_size
        num_patches_y = height // patch_size
        
        # Extract patches
        for i in range(num_patches_y): # height 
            for j in range(num_patches_x): # width
                
                # define the box to cut out
                box = (j * patch_size, i * patch_size, (j + 1) * patch_size, (i + 1) * patch_size)
                patch = image.crop(box)
                
                patch_path = os.path.join(patch_folder, f'patch_{i}_{j}.png') # folder where the patches will be saved
                patch.save(patch_path) # save patches in the folder 
                
                # extract features
                embeddings = self.extract_embeddings(patch)
                np.save(os.path.join(embedding_folder, f"patch_{i}_{j}.npy"), embeddings) # saving patch embeddings
                

    def calculate_cosine_sim(self, ref_embedding_path, embedding_folder):
        '''
        Calculate the cosine similarity between the reference embeddings and the embeddings of the patches.
        '''
        ref_embeddings = np.load(ref_embedding_path).flatten() # load the ref img features
        
        # need to calculate the num of row and columns for creating heatmap grid
        max_rows = 0
        max_cols = 0
        
        # This is for counting the num of patches and for creating grid for simiilarity heatmap
        for filename in os.listdir(embedding_folder):
            if filename.endswith(".npy"):
                parts = filename.split('_')
                row = int(parts[-2]) # patch row index
                col = int(parts[-1].split('.')[0]) # patch column index
                
                max_rows = max(max_rows, row)
                max_cols = max(max_cols, col)
                
        similarity_grid = np.zeros((max_rows + 1, max_cols + 1), dtype=float)  # +1 because indices start at 0

        # This is for loading the similarity grid with embeddings
        for filename in os.listdir(embedding_folder):
            if filename.endswith(".npy"):
                
                parts = filename.split('_')
                row = int(parts[-2])
                col = int(parts[-1].split('.')[0])  # Patch column index
                
                # load and flatten test image features
                test_embeddings = np.load(os.path.join(embedding_folder, filename)).flatten()
                
                # calculate cosine similarity
                cosine_similarity = np.dot(ref_embeddings, test_embeddings) / (np.linalg.norm(ref_embeddings) * np.linalg.norm(test_embeddings))
                
                similarity_grid[row, col] = cosine_similarity
        return similarity_grid
    
    
    def heatmap(self, ref_embedding_path, embedding_folder, heatmap_file_path, overlay_heatmap_path):
        '''
        Create a similarity heatmap based on the cosine similarity in each patch
        '''
        
        similarity_heatmap_grid = self.calculate_cosine_sim(ref_embedding_path, embedding_folder)
        heatmap = np.array(similarity_heatmap_grid)
        colors = [(0.0, "darkgreen"),
                  (0.65, "green"),
                  (0.73, "red"),
                  (1.0, "darkred")]
        
        cmap = mcolors.LinearSegmentedColormap.from_list("custom_cmap", colors)
        
        # showing and saving similarity heatmap
        plt.figure(figsize=(10,8))
        sns.heatmap(heatmap, cmap=cmap, annot=True, fmt=".2f", cbar=True, vmin=0.65, vmax=0.72)
        plt.title('RCM Cosine Similarity Heatmap with ResNet50')
        plt.savefig(heatmap_file_path, format='png', dpi=300, bbox_inches='tight')
        plt.show()
        print("Changed the heatmap!!")
        
        # for the purpose of overlaying heatmap onto the test image
        plt.figure(figsize=(10,8))
        sns.heatmap(heatmap, cmap=cmap, annot=True, fmt=".2f", cbar=True, vmin=0.65, vmax=0.72)
        plt.xticks([])  # Remove x-axis labels
        plt.yticks([])  # Remove y-axis labels
        plt.savefig(overlay_heatmap_path, format='png', dpi=300, bbox_inches='tight')
        plt.close()  # Close the plot to free up memory 
        
        
    def overlay_heatmap(self, overlay_heatmap_path, original_image):
        '''
        Overlay the heatmap onto the original image
        '''
        
        original_image = cv2.imread(original_image)
        heatmap = cv2.imread(overlay_heatmap_path)
        
        # Resize the heatmap if necessary to match the original image size
        heatmap = cv2.resize(heatmap, (original_image.shape[1], original_image.shape[0]))
        
        overlay = original_image.copy()
        alpha = 0.5 # Adjust transparency level (0: fully transparent, 1: fully opaque)
        overlay = cv2.addWeighted(original_image, 1 - alpha, heatmap, alpha, 0)
        
        plt.title('Overlayed RCM Heatmap From ResNet50')
        plt.imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
        plt.axis('off')
        plt.savefig(overlay_heatmap_path, format='png', dpi=300, bbox_inches='tight')
        plt.show()