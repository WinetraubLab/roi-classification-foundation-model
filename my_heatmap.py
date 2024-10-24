import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
       
""" This functions picks a patch from an PIL.Image and returns it.
INPUTS:
  image: PIL.Image to get the patch from.
  patch_center_x, patch_center_y: center position in pixels.
  patch_size: number of pixels accross.
  plot_patch_over_image: for debug purposes."""
def get_patch_from_image(image, patch_center_x, patch_center_y, patch_size=256, plot_patch_over_image=False, plot_patch_over_image_color='red'):
  
  # Four edges of the patch
  half_patch_size = patch_size // 2
  top_left = (patch_center_x - half_patch_size, patch_center_y - half_patch_size)
  top_right = (patch_center_x + half_patch_size, patch_center_y - half_patch_size)
  bottom_left = (patch_center_x - half_patch_size, patch_center_y + half_patch_size)
  bottom_right = (patch_center_x + half_patch_size, patch_center_y + half_patch_size)

  # Check if the patch corners are outside the test image boundaries
  width, height = image.size # get the width and height of the test image
  if top_left[0] < 0 or top_left[1] < 0 or top_right[0] > width or top_right[1] > height or bottom_left[0] < 0 or bottom_left[1] > height or bottom_right[0] > width or bottom_right[1] > height:
    raise ValueError(f"Patch centered at ({patch_center_x},{patch_center_y}) is outside the image")

  patch = image.crop((top_left[0], top_left[1], bottom_right[0], bottom_right[1]))

  if plot_patch_over_image: 
    # Create a draw object
    image_duplicate = image.copy()
    draw = ImageDraw.Draw(image_duplicate)

    # Define the rectangle based on top-left and bottom-right coordinates
    rectangle = [top_left, bottom_right]

    # Draw the rectangle
    draw.rectangle(rectangle, outline=plot_patch_over_image_color, width=3)

    # Plot the image with the rectangle
    plt.imshow(image_duplicate)
    plt.axis('off')  # Hide axes for better visibility
    plt.show()

  return patch
