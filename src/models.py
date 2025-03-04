import torch
from diffusers.pipelines.controlnet import StableDiffusionControlNetInpaintPipeline
from diffusers import ControlNetModel, UniPCMultistepScheduler, AutoPipelineForText2Image
from diffusers import StableDiffusionXLPipeline
from transformers import AutoImageProcessor, UperNetForSemanticSegmentation, AutoModelForDepthEstimation
import gc

device = "cuda"
dtype = torch.float16

def get_segmentation_pipeline():
    """Method to load the segmentation pipeline"""
    image_processor = AutoImageProcessor.from_pretrained(
        "openmmlab/upernet-convnext-small"
    )
    image_segmentor = UperNetForSemanticSegmentation.from_pretrained(
        "openmmlab/upernet-convnext-small"
    )
    return image_processor, image_segmentor

def get_depth_pipeline():
    feature_extractor = AutoImageProcessor.from_pretrained("depth-anything/Depth-Anything-V2-Large-hf",
                                                           torch_dtype=dtype)
    depth_estimator = AutoModelForDepthEstimation.from_pretrained("depth-anything/Depth-Anything-V2-Large-hf",
                                                                  torch_dtype=dtype)
    return feature_extractor, depth_estimator

def initialize_pipelines():
    controlnet_depth = ControlNetModel.from_pretrained(
        "controlnet_depth", torch_dtype=dtype, use_safetensors=True)
    controlnet_seg = ControlNetModel.from_pretrained(
        "own_controlnet", torch_dtype=dtype, use_safetensors=True)

    pipe = StableDiffusionControlNetInpaintPipeline.from_pretrained(
        "SG161222/Realistic_Vision_V6.0_B1_noVAE",
        controlnet=[controlnet_depth, controlnet_seg],
        safety_checker=None,
        torch_dtype=dtype
    )

    pipe.load_ip_adapter("h94/IP-Adapter", subfolder="models",
                         weight_name="ip-adapter_sd15.bin")
    pipe.set_ip_adapter_scale(0.4)
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    pipe = pipe.to(device)

    guide_pipe = StableDiffusionXLPipeline.from_pretrained("segmind/SSD-1B",
                                                           torch_dtype=dtype, use_safetensors=True, variant="fp16")
    guide_pipe = guide_pipe.to(device)

    return pipe, guide_pipe

def flush():
    gc.collect()
    torch.cuda.empty_cache() 