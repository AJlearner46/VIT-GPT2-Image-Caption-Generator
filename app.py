# app.py (Content of this file should be your 'gradio_code_debugged_v2' from previous steps)
import gradio as gr
import torch
import torch.nn as nn
from transformers import ViTModel, GPT2LMHeadModel, GPT2TokenizerFast, ViTFeatureExtractor, GPT2Config
from huggingface_hub import hf_hub_download
from PIL import Image
import asyncio
import concurrent.futures

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load Model & Tokenizer
class ViT_GPT2_Captioner(nn.Module):
    def __init__(self):
        super(ViT_GPT2_Captioner, self).__init__()
        self.vit = ViTModel.from_pretrained('google/vit-base-patch16-224-in21k')
        gpt2_config = GPT2Config.from_pretrained('gpt2')
        gpt2_config.add_cross_attention = True
        self.gpt2 = GPT2LMHeadModel.from_pretrained('gpt2', config=gpt2_config)
        self.bridge = nn.Linear(self.vit.config.hidden_size, self.gpt2.config.n_embd)
        for param in self.vit.parameters():
            param.requires_grad = False

    def forward(self, pixel_values, captions, attention_mask=None):
        visual_features = self.vit(pixel_values=pixel_values).last_hidden_state
        projected_features = self.bridge(visual_features[:, 0, :])
        outputs = self.gpt2(input_ids=captions, attention_mask=attention_mask,
                                    encoder_hidden_states=projected_features.unsqueeze(1),
                                    encoder_attention_mask=torch.ones(projected_features.size(0), 1).to(projected_features.device))
        return outputs.logits

model_path = hf_hub_download(repo_id="ayushrupapara/vit-gpt2-flickr8k-image-captioner", filename="model.pth") # Correct repo_id
model = ViT_GPT2_Captioner().to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

tokenizer = GPT2TokenizerFast.from_pretrained("ayushrupapara/vit-gpt2-flickr8k-image-captioner", force_download=True) # Correct repo_id
tokenizer.pad_token = tokenizer.eos_token
feature_extractor = ViTFeatureExtractor.from_pretrained("google/vit-base-patch16-224-in21k")

import asyncio
import concurrent.futures

executor = concurrent.futures.ThreadPoolExecutor()

# beam search with tunning
async def generate_caption_async(image, num_beams, temperature): 
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(executor, generate_caption_sync, image, num_beams, temperature) 

def generate_caption_sync(image, num_beams=5, temperature=0.5, max_length=20):
    #print(f"Received max_length: {max_length}, Type: {type(max_length)}") 
    max_length = int(max_length) 
    #print(f"Max_length after int conversion: {max_length}, Type: {type(max_length)}") 


    if image is None:
        return "No image uploaded"
    if isinstance(image, Image.Image):
        image = image.convert("RGB")
    else:
        raise TypeError("Invalid image format. Expected a PIL Image.")

    pixel_values = feature_extractor(images=image, return_tensors="pt").pixel_values.to(device)

    with torch.no_grad():
        input_ids = torch.tensor([[tokenizer.eos_token_id]], device=device)
        output_ids = model.gpt2.generate( # Using model.gpt2.generate for beam search
            inputs=input_ids,
            encoder_hidden_states=model.bridge(model.vit(pixel_values=pixel_values).last_hidden_state[:, 0, :]).unsqueeze(1),
            max_length=max_length, 
            num_beams=num_beams,     
            temperature=temperature,   
            length_penalty=0.9, 
            no_repeat_ngram_size=2, 
            early_stopping=True,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    caption = tokenizer.decode(output_ids.squeeze(), skip_special_tokens=True)
    return caption


iface = gr.Interface(fn=generate_caption_async, 
                     inputs=[
                         gr.Image(type="pil"), 
                         gr.Slider(minimum=1, maximum=10, step=1, value=5, label="Number of Beams"), 
                         gr.Slider(minimum=0.1, maximum=2.0, step=0.1, value=0.7, label="Temperature")     
                     ],
                     outputs="text",
                     title="ViT-GPT2 Image Captioning", 
                     description="Upload an image to get a caption.")



iface.launch() # Removed debug=True for deployment