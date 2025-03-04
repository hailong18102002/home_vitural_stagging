import gradio as gr
from .models import flush
from .image_processing import resize_dimensions

css = """
#img-display-container {
    max-height: 50vh;
    }
#img-display-input {
    max-height: 40vh;
    }
#img-display-output {
    max-height: 40vh;
    }
"""

def create_demo(model):
    gr.Markdown("### Stable Design demo")
    with gr.Row():
        with gr.Column():
            input_image = gr.Image(label="Input Image", type='pil', elem_id='img-display-input')

            with gr.Row():
                design_style = gr.Dropdown(
                    label="Design Style",
                    choices=[
                        "Modern", "Contemporary", "Minimalist", "Industrial",
                        "Scandinavian", "Traditional", "Mid-century Modern",
                        "Bohemian", "Rustic", "Art Deco", "Coastal",
                        "French Country", "Mediterranean", "Japanese Zen"
                    ],
                    value="Modern"
                )
                room_type = gr.Dropdown(
                    label="Room Type",
                    choices=[
                        "Living Room", "Bedroom", "Kitchen", "Dining Room",
                        "Bathroom", "Home Office", "Kids Room", "Master Bedroom",
                        "Entertainment Room", "Study Room"
                    ],
                    value="Living Room"
                )
                color_mood = gr.Dropdown(
                    label="Color Mood",
                    choices=[
                        "Warm & Cozy", "Cool & Calm", "Bright & Vibrant",
                        "Neutral & Natural", "Dark & Dramatic", "Pastel & Soft",
                        "Monochromatic", "Earth Tones"
                    ],
                    value="Warm & Cozy"
                )

            input_text = gr.Textbox(label='Additional Prompt Details', placeholder='Add specific details to your design', lines=2)

            with gr.Accordion('Advanced options', open=False):
                num_steps = gr.Slider(label='Steps',
                                      minimum=1,
                                      maximum=50,
                                      value=50,
                                      step=1)
                img_size = gr.Slider(label='Image size',
                                      minimum=256,
                                      maximum=768,
                                      value=768,
                                      step=64)
                guidance_scale = gr.Slider(label='Guidance Scale',
                                           minimum=0.1,
                                           maximum=30.0,
                                           value=10.0,
                                           step=0.1)
                seed = gr.Slider(label='Seed',
                                 minimum=-1,
                                 maximum=2147483647,
                                 value=323*111,
                                 step=1,
                                 randomize=True)
                strength = gr.Slider(label='Strength',
                                           minimum=0.1,
                                           maximum=1.0,
                                           value=0.9,
                                           step=0.1)
                a_prompt = gr.Textbox(
                    label='Added Prompt',
                    value="interior design, 4K, high resolution, photorealistic, highly detailed, professional photography, natural lighting, ultra realistic")
                n_prompt = gr.Textbox(
                    label='Negative Prompt',
                    value="window, door, low resolution, banner, logo, watermark, text, deformed, blurry, out of focus, surreal, ugly, beginner, cartoon, anime, illustration, painting, drawing, art, animated, 3d render")
            submit = gr.Button("Submit")

        with gr.Column():
            design_image = gr.Image(label="Output Mask", elem_id='img-display-output')

    def on_submit(image, text, style, room, color, num_steps, guidance_scale, seed, strength, a_prompt, n_prompt, img_size):
        combined_prompt = f"A {color.lower()} {style.lower()} style {room.lower()}, {text}"

        model.seed = seed
        model.neg_prompt = n_prompt
        model.additional_quality_suffix = a_prompt

        with torch.no_grad():
            out_img = model.generate_design(
                image,
                combined_prompt,
                guidance_scale=guidance_scale,
                num_steps=num_steps,
                strength=strength,
                img_size=img_size
            )

        return out_img

    submit.click(
        on_submit,
        inputs=[
            input_image, input_text, design_style, room_type, color_mood,
            num_steps, guidance_scale, seed, strength, a_prompt, n_prompt, img_size
        ],
        outputs=design_image
    )

    examples = gr.Examples(
        examples=[
            ["imgs/bedroom_1.jpg", "with a grand king-size bed, geometric bedding, luxurious velvet armchair", "Art Deco", "Bedroom", "Warm & Cozy"],
            ["imgs/living_room_1.jpg", "with vintage teak coffee table, classic sunburst clock, shag rug", "Mid-century Modern", "Living Room", "Neutral & Natural"],
        ],
        inputs=[input_image, input_text, design_style, room_type, color_mood],
        cache_examples=False
    ) 