from segmentation import initialize_sam_model, preprocess, generate_masks
import matplotlib.pyplot as plt
import numpy as np
import cv2
import copy
import torch
from torchvision.transforms.functional import to_pil_image


def show_anns(masks, save_path=None):
    if len(masks) == 0:
        return # if no masks gen due to thresholds, return

    # Need to sort the masks based on desc order of area(IoU)
    # We need to retain the mask-id
    masks = enumerate(masks)
    masks = sorted(masks, key=(lambda x: x[1]['area']), reverse = True)
    # print(masks)

    # Set axis prop
    ax = plt.gca()

    # Disable autoscale to keep the image size consistent
    ax.set_autoscale_on(False)
    plt.axis("off")

    # Iterate through each mask and display it on top of the original image
    for original_idx, ann in masks:
        m = ann['segmentation']
        img = np.ones((m.shape[0], m.shape[1], 3))

        # Generate a random color for the mask
        color_mask = np.random.random((1, 3)).tolist()[0]
        for i in range(3):
            img[:,:,i] = color_mask[i]

        # Blend the mask with the image, using 0.35 as the alpha value for transparency
        ax.imshow(np.dstack((img, m*0.35)))

        # Find contours of the mask to compute the centroid
        contours, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            cnt = contours[0]
            M = cv2.moments(cnt)

            # Compute the centroid of the mask if the moment is non-zero
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])

                # Display the original index number (1-based) at the centroid of the mask
                # The text is white, bold, and has a font size of 16
                if(original_idx == 3):
                    ax.text(cx, cy, "skirt", color='white', fontsize=16, ha='center', va='center', fontweight='bold')
                else:
                    ax.text(cx, cy, str(original_idx), color='white', fontsize=16, ha='center', va='center', fontweight='bold')


    # If save_path is provided, save the figure
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0, dpi=300)
        print(f"Saved segmentation visualization to {save_path}")
    return

def create_image_grid(original_image, images, names, rows, columns):
    names = copy.copy(names)  # Create a copy of the names list to avoid modifying the external variable
    images = copy.copy(images)  # Create a copy of the images list to avoid modifying the external variable

    # Check if images is a tensor
    if torch.is_tensor(images):
        # Check if the number of tensor images and names is equal
        assert images.size(0) == len(names), "Number of images and names should be equal"

        # Check if there are enough images for the specified grid size
        assert images.size(0) >= (rows * columns) - 1 - 1, "Not enough images for the specified grid size"

        # Convert tensor images to PIL images and apply sigmoid normalization
        images = [to_pil_image(torch.sigmoid(img)) for img in images]
    else:
        # Check if the number of PIL images and names is equal
        assert len(images) == len(names), "Number of images and names should be equal"

    # Check if there are enough images for the specified grid size
    assert len(images) >= (rows * columns) - 1 - 1, "Not enough images for the specified grid size"

    # Add the original image to the beginning of the images list
    images.insert(0, original_image)

    # Add an empty name for the original image to the beginning of the names list
    names.insert(0, '')

    # Create a figure with specified rows and columns
    fig, axes = plt.subplots(rows, columns, figsize=(15, 15))

    # Iterate through the images and names
    for idx, (img, name) in enumerate(zip(images, names)):
        # Calculate the row and column index for the current image
        row, col = divmod(idx, columns)

        # Add the image to the grid
        axes[row, col].imshow(img, cmap='gray' if idx > 0 and torch.is_tensor(images) else None)

        # Set the title (name) for the subplot
        axes[row, col].set_title(name)

        # Turn off axes for the subplot
        axes[row, col].axis('off')

    # Iterate through unused grid cells
    for idx in range(len(images), rows * columns):
        # Calculate the row and column index for the current cell
        row, col = divmod(idx, columns)

        # Turn off axes for the unused grid cell
        axes[row, col].axis('off')

    # Adjust the subplot positions to eliminate overlaps
    plt.tight_layout()

    # Display the grid of images with their names
    # plt.show()
    plt.savefig("../data/output_images/output.png")
    return

if __name__ == "__main__":
    sam_model = initialize_sam_model()
    # print( type(sam_model) )

    ### Example image from unsplash.com
    ### Photo by Lac McGregor, Canada
    ### Free to use under the Unsplash License
    ### Link: https://unsplash.com/photos/AsJirOOLN_s
    # Get the image from input_images

    source_image, segmented_image = preprocess()
    masks = generate_masks(sam_model, segmented_image)

    # print( type(masks) )
    # print( masks )

    # First let's plot trhe segmented_image, then we can overlay the masks
    fig, ax = plt.subplots(figsize=(10, 10))  # Set a reasonable figure size
    plt.imshow(segmented_image)
    show_anns(masks, save_path = "../data/output_images/segmentation.png")
    plt.close(fig)