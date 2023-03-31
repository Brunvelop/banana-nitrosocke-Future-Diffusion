import base64
from io import BytesIO
from sd import StableDifussion2

# Init is ran on server startup
# Load your model to GPU as a global variable here using the variable name "model"
def init():
    global model
    model = StableDifussion2("nitrosocke/Future-Diffusion")

# Inference is ran for every server call
# Reference your preloaded global model variable here.
def inference(model_inputs:dict) -> dict:
    global model

    # Parse out your arguments
    prompt = "future style "+ model_inputs.get('prompt', None) +" cinematic lights, trending on artstation, avengers endgame, emotional"
    negative_prompt="duplicate heads bad anatomy extra legs text"
    height=64*6 
    width=64*6
    if prompt == None:
        return {'message': "No prompt provided"}
    
    # Run the model
    image = model.generate_image(prompt, negative_prompt, height, width)

    # Resize output and conver to base64
    image = image.resize((250, 250))
    buffered = BytesIO()
    image.save(buffered, format="JPEG")
    image_64 = str(base64.b64encode(buffered.getvalue()))[2:-1]

    # Return the results as a dictionary
    return {"image_64":image_64}
