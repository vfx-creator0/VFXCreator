#!/bin/bash
# Run the inference script
python tests/inference.py \
    --prompt Ta-da it \
    --lora_path /path/to/Ta-da/weight/ \
    --output_file outputs \
    --ref_image assets/01.jpg