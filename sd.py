import torch
from diffusers import DiffusionPipeline

print(torch.__version__)
print(torch.version.cuda)
print(torch.cuda.is_available())

class StableDifussion2():
    def __init__(self, model_id, mode=None, savety_mode=False, device="cuda"):
        self.pipe = self.load_model(model_id, mode, savety_mode, device)
    
    def load_model(self, model_id, mode=None, savety_mode=False, device="cuda"):
        pipe = DiffusionPipeline.from_pretrained(
                model_id,
                torch_dtype=torch.float16
            ).to(device)
        return pipe
    
    def _generate_latent(self, height, width, seed=None, device="cuda"):
        generator = torch.Generator(device=device)

        # Get a new random seed, store it and use it as the generator state
        if not seed:
            seed = generator.seed()
        generator = generator.manual_seed(seed)
        
        image_latent = torch.randn(
            (1, self.pipe.unet.in_channels, height // 8, width // 8),
            generator = generator,
            device = device
        )
        return image_latent.type(torch.float16)


    def generate_image(
            self, prompt, negative_prompt, height=512, width=512, seed=None,
            num_inference_steps=20, guidance_scale = 7.5,
        ):

        latent = self._generate_latent(height, width, seed)
        images = self.pipe(
                    prompt = prompt,
                    height = height,
                    width = width,
                    num_inference_steps = num_inference_steps,
                    guidance_scale = guidance_scale,
                    negative_prompt=negative_prompt,
                    num_images_per_prompt = 1,
                    return_dict=False,
                    latents = latent
                )

        return images[0][0]


# sd = StableDifussion2("nitrosocke/Future-Diffusion")
# img = sd.generate_image("robot cat")
# img