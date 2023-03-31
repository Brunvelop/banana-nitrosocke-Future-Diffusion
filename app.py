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
    prompt = model_inputs.get('prompt', None)
    if prompt == None:
        return {'message': "No prompt provided"}
    
    # Run the model
    
    image = model.generate_image(prompt)

    buffered = BytesIO()
    image.save(buffered, format="JPEG")
    img_str = str(base64.b64encode(buffered.getvalue()))

    # Return the results as a dictionary
    return {"img_str":img_str}
