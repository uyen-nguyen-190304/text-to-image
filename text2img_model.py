import torch
from diffusers import StableDiffusionPipeline

# Parameters definition
rand_seed = torch.manual_seed(42)   # Set the random seed for reproducibility
NUM_INFERENCE_STEPS = 25            # Number of steps for the inference process
GUIDANCE_SCALE = 0.75               # Guidance scale for generating images
HEIGHT = 512                        # Height of the generated image
WIDTH = 512                         # Width of the generated image

# List of available models
model_list = [
    "nota-ai/bk-sdm-small",
    "CompVis/stable-diffusion-v1-4",
    "runwayml/stable-diffusion-v1-5",
    "hakurei/waifu-diffusion",
    "stabilityai/stable-diffusion-2-1",
    "dreamlike-art/dreamlike-photoreal-2.0"
]

def create_pipeline(model_name=model_list[0]):
    """
    Create and return a Stable Diffusion pipeline using the specified model.
    Parameters:
    model_name (str): The name of the model to use for the pipeline. Defaults to the first model in the model_list.
    Returns:
    StableDiffusionPipeline: A configured Stable Diffusion pipeline.
    """
    # Check for available hardware and set the pipeline accordingly
    if torch.cuda.is_available():
        print("Using GPU")
        pipeline = StableDiffusionPipeline.from_pretrained(
            model_name,
            torch_dtype = torch.float16,
            use_safetensors = True
        ).to("cuda")
    elif torch.backends.mps.is_available():
        print("Using MPS")
        pipeline = StableDiffusionPipeline.from_pretrained(
            model_name,
            torch_dtype = torch.float16,
            use_safetensors = True
        ).to("mps")
    else:
        print("Using CPU")
        pipeline = StableDiffusionPipeline.from_pretrained(
            model_name,
            torch_dtype = torch.float32,
            use_safetensors = True
        )
    return pipeline

def text2img(prompt, pipeline):
    """
    Generate an image from a text prompt using the given pipeline.
    Parameters:
    prompt (str): The text prompt to generate the image from.
    pipeline (StableDiffusionPipeline): The pipeline to use for image generation.
    Returns:
    PIL.Image: The generated image.
    """
    # Create a generator with the same seed for reproducibility
    generator = torch.Generator().manual_seed(42)
    # Generate the image using the pipeline
    images = pipeline(
        prompt,
        guidance_scale = GUIDANCE_SCALE,
        num_inference_steps = NUM_INFERENCE_STEPS, 
        generator = generator,
        num_images_per_prompt = 1,  
        height = HEIGHT,
        width = WIDTH
    )
    return images.images[0]  # Return the first image from the result

# Example usage
if __name__ == "__main__":
    pipeline = create_pipeline()                # Create the pipeline with the default model
    prompt = "A beautiful landscape painting"   # Define the text prompt
    image = text2img(prompt, pipeline)          # Generate the image from the prompt
    image.save("output.png")                    # Save the generated image to a file