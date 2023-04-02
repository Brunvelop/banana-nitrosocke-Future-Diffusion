import torch
import base64
from io import BytesIO
from diffusers import DiffusionPipeline
# Init is ran on server startup
# Load your model to GPU as a global variable here using the variable name "model"
def init():
    global model
    model = DiffusionPipeline.from_pretrained(
        "nitrosocke/Future-Diffusion",
        torch_dtype=torch.float16
    ).to('cuda')


def _generate_latent(height, width, seed=None, device="cuda"):
    generator = torch.Generator(device=device)

    # Get a new random seed, store it and use it as the generator state
    if not seed:
        seed = generator.seed()
    generator = generator.manual_seed(seed)
    
    image_latent = torch.randn(
        (1, model.unet.in_channels, height // 8, width // 8),
        generator = generator,
        device = device
    )
    return image_latent.type(torch.float16)
    
# Inference is ran for every server call
# Reference your preloaded global model variable here.
def inference(model_inputs:dict) -> dict:
    global model

    latent = _generate_latent(64*6, 64*6)
    images = model(
        prompt = "future style "+ model_inputs.get('prompt', None) +" cinematic lights, trending on artstation, avengers endgame, emotional",
        height=64*7,
        width=64*7,
        num_inference_steps = 20,
        guidance_scale = 7.5,
        negative_prompt="duplicate heads bad anatomy extra legs text",
        num_images_per_prompt = 1,
        return_dict=False,
        latents = latent
    )
    image = images[0][0]
    
    # Resize output and conver to base64
    # image = image.resize((250, 250))
    buffered = BytesIO()
    image.save(buffered, format="JPEG")
    image_base64 = str(base64.b64encode(buffered.getvalue()))[2:-1]

    # Return the results as a dictionary
    return {"image_base64":image_base64}
