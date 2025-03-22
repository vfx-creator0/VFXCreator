"""
Run this test in Lora adpater checking:

```shell
python3 test_lora_inference.py --prompt "A girl is ridding a bike." --model_path "THUDM/CogVideoX-5B" --lora_path "path/to/lora" --lora_name "lora_adapter" --output_file "output.mp4" --fps 8
```

"""
import os
import argparse
import torch
from diffusers import CogVideoXPipeline
from diffusers.utils import export_to_video
from diffusers import (
    AutoencoderKLCogVideoX,
    CogVideoXDPMScheduler,
    CogVideoXImageToVideoPipeline,
    CogVideoXTransformer3DModel,
)
import numpy as np
import random
from diffusers.utils import convert_unet_state_dict_to_peft, export_to_video, load_image

def generate_video(model_path,ref_image, prompt, lora_path, lora_name, output_file, fps):

    pipe = CogVideoXImageToVideoPipeline.from_pretrained(model_path,torch_dtype=torch.bfloat16).to("cuda")
    pipe.load_lora_weights(lora_path, weight_name="pytorch_lora_weights.safetensors", adapter_name=lora_name)
    pipe.set_adapters([lora_name], [1.0])
    pipe.enable_model_cpu_offload()
    pipe.vae.enable_slicing()
    pipe.vae.enable_tiling()
    steps=lora_path.split('/')[-1].split('-')[-1]
    video = pipe(image=load_image(ref_image),prompt=prompt).frames[0]
    export_to_video(video, output_file, fps=fps)


def main():
    parser = argparse.ArgumentParser(description="Generate video using CogVideoX and LoRA weights")
    parser.add_argument("--prompt", type=str,default="")
    parser.add_argument("--model_path", type=str, default="THUDM/CogVideoX-5b-I2V", help="Base Model path or HF ID")
    parser.add_argument("--lora_path", type=str, default="")
    parser.add_argument("--lora_name", type=str, default="lora_adapter", help="Name of the LoRA adapter")
    parser.add_argument("--output_file", type=str, default="output.mp4", help="Output video file name")
    parser.add_argument("--ref_image", type=str, default="")

    args = parser.parse_args()

    generate_video(args.model_path,args.ref_image,args.prompt, args.lora_path, args.lora_name, args.output_file, args.fps)


if __name__ == "__main__":
    main()