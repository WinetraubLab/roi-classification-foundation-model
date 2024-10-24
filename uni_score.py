import numpy as np
import os
from PIL import Image
import timm
import torch
from torchvision import transforms

""" Use this class to compute score between two patches """
class UNIScore:
  """ Set up environment """
  def __init__(self, uni_weights_file_path = "/content/drive/Shareddrives/Yolab - Current Projects/Kyi Lei Aye/UNI_Model/pytorch_model.bin"):

    # Check if we already downloaded the weights
    if not os.path.isfile(uni_weights_file_path):
      raise FileNotFoundError(f"We are unable to locate the uni_weights_file_path. Please use the following steps:\n" + 
                              "1. Login to huggingface with a HF token with gated access permission.\n" +
                              "2. hf_hub_download the weights.\n" + 
                              "To do so. Type the following commands to the colab code:\n\n" + 
                              "!huggingface-cli login\n" +
                              "from huggingface_hub import hf_hub_download\n" +
                              'hf_hub_download("MahmoodLab/UNI", filename="pytorch_model.bin", local_dir="/content/drive/Shareddrives/Yolab - Current Projects/Kyi Lei Aye/UNI_Model/", force_download=True)')

    # Create the model
    self.model = timm.create_model(
        "vit_large_patch16_224", img_size=224, patch_size=16, init_values=1e-5, num_classes=0, dynamic_img_size=True
    )

    # Load the downloaded model weights
    self.model.load_state_dict(torch.load(uni_weights_file_path, map_location="cpu"), strict=True)

    # Define image transform
    self.transform = transforms.Compose(
        [
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )

    # Set the model to evaluation mode
    self.model.eval()

  """ This function computes the score between two patches. 
  Both patches should be a PIL.Image class """
  def compute_similarity_between_patches(self, patch_a, patch_b, patch_size=256):

    # Check patch sizes
    def check_patch_size(p):
      width, height = p.size
      if width != patch_size or height != patch_size:
        raise ValueError(f"patch_a and patch_b should be of size {patch_size}")
    check_patch_size(patch_a)
    check_patch_size(patch_b)

    # Make sure PIL.image is RGB
    patch_a = patch_a.convert("RGB")
    patch_b = patch_b.convert("RGB")

    # Transform image patches
    patch_a_t = self.transform(patch_a).unsqueeze(dim=0) # transform test image patch
    patch_b_t = self.transform(patch_b).unsqueeze(dim=0) # transform test image patch

    # Compute feature embeddings
    with torch.inference_mode():
      patch_a_emb = self.model(patch_a_t) 
      patch_b_emb = self.model(patch_b_t)

    # Return cosine similarity
    cosine_similarity = np.dot(patch_a_emb, patch_b_emb) / (np.linalg.norm(patch_a_emb) * np.linalg.norm(patch_b_emb))
    return cosine_similarity
       
