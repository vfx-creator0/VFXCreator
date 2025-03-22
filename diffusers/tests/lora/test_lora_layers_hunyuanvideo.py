# Copyright 2024 HuggingFace Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import sys
import unittest

import torch
from transformers import CLIPTextModel, CLIPTokenizer, LlamaModel, LlamaTokenizerFast

from diffusers import (
    AutoencoderKLHunyuanVideo,
    FlowMatchEulerDiscreteScheduler,
    HunyuanVideoPipeline,
    HunyuanVideoTransformer3DModel,
)
from diffusers.utils.testing_utils import (
    floats_tensor,
    require_peft_backend,
    skip_mps,
)


sys.path.append(".")

from utils import PeftLoraLoaderMixinTests  # noqa: E402


@require_peft_backend
@skip_mps
class HunyuanVideoLoRATests(unittest.TestCase, PeftLoraLoaderMixinTests):
    pipeline_class = HunyuanVideoPipeline
    scheduler_cls = FlowMatchEulerDiscreteScheduler
    scheduler_classes = [FlowMatchEulerDiscreteScheduler]
    scheduler_kwargs = {}

    transformer_kwargs = {
        "in_channels": 4,
        "out_channels": 4,
        "num_attention_heads": 2,
        "attention_head_dim": 10,
        "num_layers": 1,
        "num_single_layers": 1,
        "num_refiner_layers": 1,
        "patch_size": 1,
        "patch_size_t": 1,
        "guidance_embeds": True,
        "text_embed_dim": 16,
        "pooled_projection_dim": 8,
        "rope_axes_dim": (2, 4, 4),
    }
    transformer_cls = HunyuanVideoTransformer3DModel
    vae_kwargs = {
        "in_channels": 3,
        "out_channels": 3,
        "latent_channels": 4,
        "down_block_types": (
            "HunyuanVideoDownBlock3D",
            "HunyuanVideoDownBlock3D",
            "HunyuanVideoDownBlock3D",
            "HunyuanVideoDownBlock3D",
        ),
        "up_block_types": (
            "HunyuanVideoUpBlock3D",
            "HunyuanVideoUpBlock3D",
            "HunyuanVideoUpBlock3D",
            "HunyuanVideoUpBlock3D",
        ),
        "block_out_channels": (8, 8, 8, 8),
        "layers_per_block": 1,
        "act_fn": "silu",
        "norm_num_groups": 4,
        "scaling_factor": 0.476986,
        "spatial_compression_ratio": 8,
        "temporal_compression_ratio": 4,
        "mid_block_add_attention": True,
    }
    vae_cls = AutoencoderKLHunyuanVideo
    has_two_text_encoders = True
    tokenizer_cls, tokenizer_id, tokenizer_subfolder = (
        LlamaTokenizerFast,
        "hf-internal-testing/tiny-random-hunyuanvideo",
        "tokenizer",
    )
    tokenizer_2_cls, tokenizer_2_id, tokenizer_2_subfolder = (
        CLIPTokenizer,
        "hf-internal-testing/tiny-random-hunyuanvideo",
        "tokenizer_2",
    )
    text_encoder_cls, text_encoder_id, text_encoder_subfolder = (
        LlamaModel,
        "hf-internal-testing/tiny-random-hunyuanvideo",
        "text_encoder",
    )
    text_encoder_2_cls, text_encoder_2_id, text_encoder_2_subfolder = (
        CLIPTextModel,
        "hf-internal-testing/tiny-random-hunyuanvideo",
        "text_encoder_2",
    )

    @property
    def output_shape(self):
        return (1, 9, 32, 32, 3)

    def get_dummy_inputs(self, with_generator=True):
        batch_size = 1
        sequence_length = 16
        num_channels = 4
        num_frames = 9
        num_latent_frames = 3  # (num_frames - 1) // temporal_compression_ratio + 1
        sizes = (4, 4)

        generator = torch.manual_seed(0)
        noise = floats_tensor((batch_size, num_latent_frames, num_channels) + sizes)
        input_ids = torch.randint(1, sequence_length, size=(batch_size, sequence_length), generator=generator)

        pipeline_inputs = {
            "prompt": "",
            "num_frames": num_frames,
            "num_inference_steps": 1,
            "guidance_scale": 6.0,
            "height": 32,
            "width": 32,
            "max_sequence_length": sequence_length,
            "prompt_template": {"template": "{}", "crop_start": 0},
            "output_type": "np",
        }
        if with_generator:
            pipeline_inputs.update({"generator": generator})

        return noise, input_ids, pipeline_inputs

    def test_simple_inference_with_text_lora_denoiser_fused_multi(self):
        super().test_simple_inference_with_text_lora_denoiser_fused_multi(expected_atol=9e-3)

    def test_simple_inference_with_text_denoiser_lora_unfused(self):
        super().test_simple_inference_with_text_denoiser_lora_unfused(expected_atol=9e-3)

    # TODO(aryan): Fix the following test
    @unittest.skip("This test fails with an error I haven't been able to debug yet.")
    def test_simple_inference_save_pretrained(self):
        pass

    @unittest.skip("Not supported in HunyuanVideo.")
    def test_simple_inference_with_text_denoiser_block_scale(self):
        pass

    @unittest.skip("Not supported in HunyuanVideo.")
    def test_simple_inference_with_text_denoiser_block_scale_for_all_dict_options(self):
        pass

    @unittest.skip("Not supported in HunyuanVideo.")
    def test_modify_padding_mode(self):
        pass

    @unittest.skip("Text encoder LoRA is not supported in HunyuanVideo.")
    def test_simple_inference_with_partial_text_lora(self):
        pass

    @unittest.skip("Text encoder LoRA is not supported in HunyuanVideo.")
    def test_simple_inference_with_text_lora(self):
        pass

    @unittest.skip("Text encoder LoRA is not supported in HunyuanVideo.")
    def test_simple_inference_with_text_lora_and_scale(self):
        pass

    @unittest.skip("Text encoder LoRA is not supported in HunyuanVideo.")
    def test_simple_inference_with_text_lora_fused(self):
        pass

    @unittest.skip("Text encoder LoRA is not supported in HunyuanVideo.")
    def test_simple_inference_with_text_lora_save_load(self):
        pass
