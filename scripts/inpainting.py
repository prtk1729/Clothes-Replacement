'''
Idea: Just for my won clarity on what to code here.
    - Image -> SAMMaskGenerator() -> Mask + Image -> [Here, I am now]
    - Pick a part. mask -> Get the pixel coords of this mask
    - Using SD inpainting, we can guide this area using my prompt, to inpaint.
'''

from diffusers import EulerDiscreteScheduler # to load the pretrained scheduler for denoising
from diffusers import StableDiffusionInpaintPipeline # load the pretrained SD model for inpainting task
import torch
from segmentation import initialize_sam_model, generate_masks, preprocess
from config import *
from PIL import Image
from torchvision.transforms.functional import to_pil_image

from visualisation import create_image_grid

def load_inpainting_pipeline(model_dir = INPAINT_MODEL_DIR):
    ''' Load and set the pipeline '''

    # load the scheduler
    scheduler = EulerDiscreteScheduler.from_pretrained( pretrained_model_name_or_path=model_dir, \
                                                       subfolder = "scheduler" ) 
    # print( type(scheduler) )

    # load the pipeline and set the scheduler
    pipeline = StableDiffusionInpaintPipeline.from_pretrained( pretrained_model_name_or_path = model_dir,
                                                               scheduler = scheduler,
                                                               revision = "fp16", # for memory issues
                                                               torch_dtype = torch.float16
                                                             )
    pipeline.to(DEVICE)

    # Other mem optimisation => xformer
    # Just for inference, init of xformers for training isn't guaranteed!
    # techniques like slicing etc makes it memory eff
    # pipeline.enable_xformers_memory_efficient_attention()
    return pipeline


def apply_inpainting(pipeline, image, mask_image, prompt, 
                     guidance_scale = 7.5, steps = 60, seed = 77):
    """  
        1. Get the particular mask: (The `skirt` mask was indexed 0), as we saw in the visualisation.py
        2. Guide the model, how much to attend to the prompt 
        3. More, the inference steps, better the inpainting result
    """
    generator = torch.Generator(device = DEVICE).manual_seed(seed)

    images = pipeline( 
                      image = image, 
                      mask_image = mask_image,
                      prompt = prompt,
                      guidance_scale = guidance_scale,
                      num_inference_steps = steps,
                      generator = generator # for reproducible res
                      )

    return images[0]


if __name__ == "__main__":
    # segmentation model
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


    pipeline = load_inpainting_pipeline()  
    
    # mask of the skirt
    prompts = ["a skirt full of text",  "red flowers", "blue flowers", "a zebra skirt"]
    skirt_index = 3
    mask_image = masks[ skirt_index ]["segmentation"] # True for pixels pred as skirt by sam else F
    mask_image = Image.fromarray(mask_image) # bool to PIL

    mask_image.save("../data/output_images/mask_image.png")

    # print( mask_image )

    encoded_images = []
    for i, prompt in enumerate(prompts):
        res = apply_inpainting( pipeline, source_image, mask_image, prompt )
        inpainted_image = res[0]
        # print(type(inpainted_image))
        encoded_images.append(inpainted_image)
    # inpainted_image = to_pil_image(torch.sigmoid(inpainted_image))
    # Display the result
    # inpainted_image.show()

    # Save the result
    # print( type(inpainted_image), len(inpainted_image) )
    # print( inpainted_image[0] )
    # inpainted_image[0].save(f"../data/output_images/{prompt}.png")

    # print( type(source_image), type(encoded_images[0]) )
    create_image_grid(source_image, encoded_images, prompts, 2, 3)
