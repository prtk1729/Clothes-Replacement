from segmentation import initialize_sam_model, preprocess, generate_masks
import matplotlib.pyplot as plt
import numpy as np
import cv2

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
                ax.text(cx, cy, str(original_idx), color='white', fontsize=16, ha='center', va='center', fontweight='bold')

    # If save_path is provided, save the figure
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0, dpi=300)
        print(f"Saved segmentation visualization to {save_path}")
    return

if __name__ == "__main__":
    sam_model = initialize_sam_model()
    # print( type(sam_model) )

    ### Example image from unsplash.com
    ### Photo by Lac McGregor, Canada
    ### Free to use under the Unsplash License
    ### Link: https://unsplash.com/photos/AsJirOOLN_s
    # Get the image from input_images

    image = preprocess()
    masks = generate_masks(sam_model, image)

    # print( type(masks) )
    # print( masks )

    # First let's plot trhe image, then we can overlay the masks
    fig, ax = plt.subplots(figsize=(10, 10))  # Set a reasonable figure size
    plt.imshow(image)
    show_anns(masks, save_path = "../data/output_images/segmentation.png")
    plt.close(fig)