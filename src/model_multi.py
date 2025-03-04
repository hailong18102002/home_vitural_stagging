import torch
import numpy as np
from PIL import Image
from models import flush
from image_processing import filter_items,segment_image, get_depth_image, resize_dimensions
from utils import map_colors_rgb
from models import initialize_pipelines, get_segmentation_pipeline, get_depth_pipeline
class ControlNetDepthDesignModelMulti:
    """ Produces random noise images """

    def __init__(self):
        """ Initialize your model(s) here """
        self.seed = 323*111
        self.neg_prompt = "window, door, low resolution, banner, logo, watermark, text, deformed, blurry, out of focus, surreal, ugly, beginner, cartoon, anime, illustration, painting, drawing, art, animated, 3d render"
        self.control_items = ["windowpane;window", "door;double;door"]
        self.additional_quality_suffix = "interior design, 4K, high resolution, photorealistic, highly detailed, professional photography, natural lighting, ultra realistic"
        self.pipe, self.guide_pipe = initialize_pipelines()

    # Initialize segmentation and depth models
        self.seg_image_processor, self.image_segmentor = get_segmentation_pipeline()
        self.depth_feature_extractor, self.depth_estimator = get_depth_pipeline()
        self.depth_estimator = self.depth_estimator.to("cuda")
    def generate_design(self, empty_room_image: Image, prompt: str, guidance_scale: int = 10, num_steps: int = 50, strength: float =0.9, img_size: int = 640) -> Image:
        """
        Given an image of an empty room and a prompt
        generate the designed room according to the prompt
        """
        print(prompt)
        flush()
        self.generator = torch.Generator(device="cuda").manual_seed(self.seed)

        pos_prompt = prompt + f', {self.additional_quality_suffix}'

        orig_w, orig_h = empty_room_image.size
        new_width, new_height = resize_dimensions(empty_room_image.size, img_size)
        input_image = empty_room_image.resize((new_width, new_height))
        real_seg = np.array(segment_image(input_image,
                                          self.seg_image_processor,
                                          self.image_segmentor))
        unique_colors = np.unique(real_seg.reshape(-1, real_seg.shape[2]), axis=0)
        unique_colors = [tuple(color) for color in unique_colors]
        segment_items = [map_colors_rgb(i) for i in unique_colors]
        chosen_colors, segment_items = filter_items(
            colors_list=unique_colors,
            items_list=segment_items,
            items_to_remove=self.control_items
        )
        mask = np.zeros_like(real_seg)
        for color in chosen_colors:
            color_matches = (real_seg == color).all(axis=2)
            mask[color_matches] = 1

        image_np = np.array(input_image)
        image = Image.fromarray(image_np).convert("RGB")
        mask_image = Image.fromarray((mask * 255).astype(np.uint8)).convert("RGB")
        segmentation_cond_image = Image.fromarray(real_seg).convert("RGB")

        image_depth = get_depth_image(image, self.depth_feature_extractor, self.depth_estimator)

        flush()
        new_width_ip = int(new_width / 8) * 8
        new_height_ip = int(new_height / 8) * 8
        ip_image = self.guide_pipe(pos_prompt,
                                   num_inference_steps=num_steps,
                                   negative_prompt=self.neg_prompt,
                                   height=new_height_ip,
                                   width=new_width_ip,
                                   generator=[self.generator]).images[0]

        flush()
        generated_image = self.pipe(
            prompt=pos_prompt,
            negative_prompt=self.neg_prompt,
            num_inference_steps=num_steps,
            strength=strength,
            guidance_scale=guidance_scale,
            generator=[self.generator],
            image=image,
            mask_image=mask_image,
            ip_adapter_image=ip_image,
            control_image=[image_depth, segmentation_cond_image],
            controlnet_conditioning_scale=[0.5, 0.5]
        ).images[0]

        flush()
        design_image = generated_image.resize(
            (orig_w, orig_h), Image.Resampling.LANCZOS
        )

        return design_image 