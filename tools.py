import replicate
import json
import os
from PIL import Image
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

"""Function to generate image"""


def generate_img(original_cloth_img_url: str, prompt: str) -> str:
    print("...STARTING TO GENERATE...")

    # Check if the URL is accessible and if the file exists
    if not original_cloth_img_url.startswith("http"):
        print(f"Error: The image URL {original_cloth_img_url} is not a valid URL.")
        return {"error": f"The image URL {original_cloth_img_url} is not a valid URL."}
    
    # Check if the environment variable is set
    if 'REPLICATE_API_TOKEN' not in os.environ:
        print("Error: REPLICATE_API_TOKEN is not set.")
        return {"error": "REPLICATE_API_TOKEN is not set."}
    

    # Initialize the Replicate client
    client = replicate.Client(api_token=os.environ['REPLICATE_API_TOKEN'])
    
    data = {
        "3": {
            "inputs": {
                "seed": 575276766819853,
                "steps": 35,
                "cfg": 8,
                "sampler_name": "dpmpp_2m",
                "scheduler": "karras",
                "denoise": 1,
                "model": ["4", 0],
                "positive": ["6", 0],
                "negative": ["7", 0],
                "latent_image": ["5", 0],
            },
            "class_type": "KSampler",
            "_meta": {"title": "KSampler"},
        },
        "4": {
            "inputs": {
                "ckpt_name": "Juggernaut-XL_v9_RunDiffusionPhoto_v2.safetensors"
            },
            "class_type": "CheckpointLoaderSimple",
            "_meta": {"title": "Load Checkpoint"},
        },
        "5": {
            "inputs": {"width": 1024, "height": 1024, "batch_size": 1},
            "class_type": "EmptyLatentImage",
            "_meta": {"title": "Empty Latent Image"},
        },
        "6": {
            "inputs": {
                "text": "hyperdetailed photography, soft light, head portrait, (white background:1.3), skin details, sharp and in focus, girl chinese student, short (red: 1.4) wavey hair, big eyes, narrow nose, slim, cute, beautiful",
                "clip": ["4", 1],
            },
            "class_type": "CLIPTextEncode",
            "_meta": {"title": "CLIP Text Encode (Prompt)"},
        },
        "7": {
            "inputs": {
                "text": "(worst quality, too close, low quality, normal quality, lowres, low details, oversaturated, undersaturated, overexposed, underexposed, grayscale, bw, bad photo, bad photography, bad art:1.4), (watermark, signature, text font, username, error, logo, words, letters, digits, autograph, trademark, name:1.2), (blur, blurry, grainy), morbid, ugly, asymmetrical, mutated malformed, mutilated, poorly lit, bad shadow, draft, cropped, out of frame, cut off, censored, jpeg artifacts, out of focus, glitch, duplicate, (airbrushed, cartoon, anime, semi-realistic, cgi, render, blender, digital art, manga, amateur:1.3), (3D, 3D Game, 3D Game Scene, 3D Character:1.1), acne\n",
                "clip": ["4", 1],
            },
            "class_type": "CLIPTextEncode",
            "_meta": {"title": "CLIP Text Encode (Prompt)"},
        },
        "8": {
            "inputs": {"samples": ["3", 0], "vae": ["4", 2]},
            "class_type": "VAEDecode",
            "_meta": {"title": "VAE Decode"},
        },
        "12": {
            "inputs": {"resolution": 1024, "image": ["13", 0]},
            "class_type": "OneFormer-COCO-SemSegPreprocessor",
            "_meta": {"title": "OneFormer COCO Segmentor"},
        },
        "13": {
            "inputs": {"image": "$13-0", "images": ["8", 0]},
            "class_type": "PreviewBridge",
            "_meta": {"title": "Preview Bridge (Image)"},
        },
        "14": {
            "inputs": {"image": "$14-0", "images": ["12", 0]},
            "class_type": "PreviewBridge",
            "_meta": {"title": "Preview Bridge (Image)"},
        },
        "15": {
            "inputs": {"channel": "red", "image": ["12", 0]},
            "class_type": "ImageToMask",
            "_meta": {"title": "Convert Image to Mask"},
        },
        "19": {
            "inputs": {
                "weight": 0.5,
                "weight_faceidv2": 1.5,
                "weight_type": "ease in",
                "combine_embeds": "concat",
                "start_at": 0,
                "end_at": 1,
                "embeds_scaling": "V only",
                "model": ["4", 0],
                "ipadapter": ["30", 0],
                "image": ["47", 0],
                "attn_mask": ["15", 0],
                "clip_vision": ["35", 0],
                "insightface": ["36", 0],
            },
            "class_type": "IPAdapterFaceID",
            "_meta": {"title": "IPAdapter FaceID"},
        },
        "30": {
            "inputs": {"ipadapter_file": "ip-adapter-faceid-plusv2_sdxl.bin"},
            "class_type": "IPAdapterModelLoader",
            "_meta": {"title": "IPAdapter Model Loader"},
        },
        "35": {
            "inputs": {"clip_name": "CLIP-ViT-H-14-laion2B-s32B-b79K.safetensors"},
            "class_type": "CLIPVisionLoader",
            "_meta": {"title": "Load CLIP Vision"},
        },
        "36": {
            "inputs": {"provider": "CUDA"},
            "class_type": "IPAdapterInsightFaceLoader",
            "_meta": {"title": "IPAdapter InsightFace Loader"},
        },
        "39": {
            "inputs": {"text": prompt, "clip": ["4", 1]},
            "class_type": "CLIPTextEncode",
            "_meta": {"title": "CLIP Text Encode (Prompt)"},
        },
        "40": {
            "inputs": {
                "text": "(worst quality, low quality, normal quality, lowres, low details, oversaturated, undersaturated, overexposed, underexposed, grayscale, bw, bad photo, bad photography, bad art:1.4), (watermark, signature, text font, username, error, logo, words, letters, digits, autograph, trademark, name:1.2), (blur, blurry, grainy), morbid, ugly, asymmetrical, mutated malformed, mutilated, poorly lit, bad shadow, draft, cropped, out of frame, cut off, censored, jpeg artifacts, out of focus, glitch, duplicate, (airbrushed, cartoon, anime, semi-realistic, cgi, render, blender, digital art, manga, amateur:1.3), (3D, 3D Game, 3D Game Scene, 3D Character:1.1), acne\n",
                "clip": ["4", 1],
            },
            "class_type": "CLIPTextEncode",
            "_meta": {"title": "CLIP Text Encode (Prompt)"},
        },
        "41": {
            "inputs": {
                "seed": 164892901598143,
                "steps": 35,
                "cfg": 8,
                "sampler_name": "dpmpp_2m",
                "scheduler": "karras",
                "denoise": 1,
                "model": ["58", 0],
                "positive": ["39", 0],
                "negative": ["40", 0],
                "latent_image": ["42", 0],
            },
            "class_type": "KSampler",
            "_meta": {"title": "KSampler"},
        },
        "42": {
            "inputs": {"width": 1024, "height": 1024, "batch_size": 1},
            "class_type": "EmptyLatentImage",
            "_meta": {"title": "Empty Latent Image"},
        },
        "43": {
            "inputs": {"samples": ["41", 0], "vae": ["4", 2]},
            "class_type": "VAEDecode",
            "_meta": {"title": "VAE Decode"},
        },
        "45": {
            "inputs": {"image": "$45-0", "images": ["43", 0]},
            "class_type": "PreviewBridge",
            "_meta": {"title": "Preview Bridge (Image)"},
        },
        "47": {
            "inputs": {"image": "$47-0", "images": ["13", 0]},
            "class_type": "PreviewBridge",
            "_meta": {"title": "Preview Bridge (Image)"},
        },
        "48": {
            "inputs": {"image": original_cloth_img_url, "upload": "image"},
            "class_type": "LoadImage",
            "_meta": {"title": "Load Image"},
        },
        "49": {
            "inputs": {"resolution": 1024, "image": ["48", 0]},
            "class_type": "UniFormer-SemSegPreprocessor",
            "_meta": {"title": "UniFormer Segmentor"},
        },
        "50": {
            "inputs": {"channel": "red", "image": ["49", 0]},
            "class_type": "ImageToMask",
            "_meta": {"title": "Convert Image to Mask"},
        },
        "54": {
            "inputs": {"images": ["49", 0]},
            "class_type": "PreviewImage",
            "_meta": {"title": "Preview Image"},
        },
        "58": {
            "inputs": {
                "weight": 0.5,
                "weight_type": "linear",
                "combine_embeds": "concat",
                "start_at": 0,
                "end_at": 1,
                "embeds_scaling": "V only",
                "model": ["61", 0],
                "ipadapter": ["59", 0],
                "image": ["48", 0],
                "attn_mask": ["50", 0],
                "clip_vision": ["60", 0],
            },
            "class_type": "IPAdapterAdvanced",
            "_meta": {"title": "IPAdapter Advanced"},
        },
        "59": {
            "inputs": {"ipadapter_file": "ip-adapter-plus_sdxl_vit-h.safetensors"},
            "class_type": "IPAdapterModelLoader",
            "_meta": {"title": "IPAdapter Model Loader"},
        },
        "60": {
            "inputs": {"clip_name": "CLIP-ViT-H-14-laion2B-s32B-b79K.safetensors"},
            "class_type": "CLIPVisionLoader",
            "_meta": {"title": "Load CLIP Vision"},
        },
        "61": {
            "inputs": {
                "weight": 0.3,
                "weight_type": "linear",
                "combine_embeds": "concat",
                "start_at": 0,
                "end_at": 1,
                "embeds_scaling": "V only",
                "model": ["19", 0],
                "ipadapter": ["62", 0],
                "image": ["47", 0],
                "attn_mask": ["15", 0],
                "clip_vision": ["35", 0],
            },
            "class_type": "IPAdapterAdvanced",
            "_meta": {"title": "IPAdapter Advanced"},
        },
        "62": {
            "inputs": {"ipadapter_file": "ip-adapter-plus-face_sdxl_vit-h.safetensors"},
            "class_type": "IPAdapterModelLoader",
            "_meta": {"title": "IPAdapter Model Loader"},
        },
        "64": {
            "inputs": {"filename_prefix": "ComfyUI", "images": ["45", 0]},
            "class_type": "SaveImage",
            "_meta": {"title": "Save Image"},
        },
    }

    json_string = json.dumps(data, indent=2)
    #print("Workflow JSON prepared:", json_string)

    try:
        output = client.run(
            "fofr/any-comfyui-workflow:cd385285ba75685a040afbded7b79814a971f3febf46c5eab7c716e200c784e1",
            input={
                "workflow_json": json_string,
                "randomise_seeds": True,
                "return_temp_files": False,
            }
        )
        
        if output:
            print("Output received:", output)
        else:
            print("No output received.")
            return None
        return {"original_img_url": original_cloth_img_url, "prompt": prompt, "output_image": output}
    except Exception as e:
        print(f"Error: Failed to execute the workflow. Exception: {e}")
        return {"error": f"Failed to execute the workflow. Exception: {e}"}



"""Function to review generated image"""
def review_img(original_cloth_img_url: str, ai_generated_img_url: str, text_prompt: str) -> str:
    print("...STARTING TO REVIEW...")
    client = OpenAI()
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": f"""
                        original text prompt: {text_prompt}
                        First image is the image of a garment, second image is an AI generated image of a person wearing that same garment; 
                        Please compare and classify if the garment in the second image
                        is highly similar, around 95% match with the first image;
                        if more than 95% match, just return "95% match",
                        if less than 95% match, list out the discrepancy,
                        and iterate the original text prompt for the image generation model to fill the gap of discrepancy
                        (just add/tweak details to the original prompt, do not do major structure changes);

                        Return in specific format:
                        MATCH SCORE: xxx,
                        CRITIQUE: xxx,
                        ITERATED PROMPT: xxx
                        """,
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": original_cloth_img_url,
                        },
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": ai_generated_img_url,
                        },
                    },
                ],
            }
        ],
        max_tokens=300,
    )
    return response.choices[0].message.content


"""Function to fix hands"""
def fix_hands(ai_generated_img_url: str) -> str:
    print("...STARTING TO FIX HANDS...")

    # Initialize the Replicate client
    client = replicate.Client(api_token=os.environ['REPLICATE_API_TOKEN'])

    output = client.run(
        "973398769/hands-restoration:721dd0c8bc13dcd514b8aa0d7530f814fc1f01caff8d5bc92b6ecc0856a7ad20",
        input={
            "input_file": ai_generated_img_url,
            "function_name": "hand_restoration",
            "randomise_seeds": True,
            "return_temp_files": False,
        },
    )
    return f"iterated image with hand distortion fixed: {output}"


"""Function to upscale image"""


def upscale_image(latest_ai_generated_img_url: str, prompt: str) -> str:
    output = replicate.run(
        "juergengunz/ultimate-portrait-upscale:f7fdace4ec7adab7fa02688a160eee8057f070ead7fbb84e0904864fd2324be5",
        input={
            "cfg": 8,
            "image": latest_ai_generated_img_url,
            "steps": 20,
            "denoise": 0.1,
            "upscaler": "4x-UltraSharp",
            "mask_blur": 8,
            "mode_type": "Linear",
            "scheduler": "normal",
            "tile_width": 512,
            "upscale_by": 2,
            "tile_height": 512,
            "sampler_name": "euler",
            "tile_padding": 32,
            "seam_fix_mode": "None",
            "seam_fix_width": 64,
            "negative_prompt": "(worst quality, too close, low quality, normal quality, lowres, low details, oversaturated, undersaturated, overexposed, underexposed, grayscale, bw, bad photo, bad photography, bad art:1.4), (watermark, signature, text font, username, error, logo, words, letters, digits, autograph, trademark, name:1.2), (blur, blurry, grainy), morbid, ugly, asymmetrical, mutated malformed, mutilated, poorly lit, bad shadow, draft, cropped, out of frame, cut off, censored, jpeg artifacts, out of focus, glitch, duplicate, (airbrushed, cartoon, anime, semi-realistic, cgi, render, blender, digital art, manga, amateur:1.3), (3D, 3D Game, 3D Game Scene, 3D Character:1.1), acne",
            "positive_prompt": prompt,
            "seam_fix_denoise": 1,
            "seam_fix_padding": 16,
            "seam_fix_mask_blur": 8,
            "controlnet_strength": 1,
            "force_uniform_tiles": True,
            "use_controlnet_tile": True,
        },
    )
    return output




if __name__ == "__main__":

    base_prompt = "a woman with red hair, wear a black jacket, in a cafe in paris"
    image_url = "https://steve-halls.github.io/autogen-virtual-try-on/jacket_2.jpeg"

    generate_img(image_url, base_prompt)


