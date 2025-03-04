import numpy as np
from typing import Tuple, Union, List
import os
from PIL import Image
import torch
import gradio as gr
from src.gui import create_demo
from src.model_multi import ControlNetDepthDesignModelMulti

device = "cuda"
dtype = torch.float16


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

def main():
    model = ControlNetDepthDesignModelMulti()
    print('Models uploaded successfully')

    title = "# StableDesign"
    description = """
    <p style='font-size: 14px; margin-bottom: 10px;'><a href='https://www.linkedin.com/in/mykola-lavreniuk/'>Mykola Lavreniuk</a>, <a href='https://www.linkedin.com/in/bartosz-ludwiczuk-a677a760/'>Bartosz Ludwiczuk</a></p>
    <p style='font-size: 16px; margin-bottom: 0px; margin-top=0px;'>Official demo for <strong>StableDesign:</strong> 2nd place solution for the Generative Interior Design 2024 <a href='https://www.aicrowd.com/challenges/generative-interior-design-challenge-2024/leaderboards?challenge_round_id=1314'>competition</a>. StableDesign is a deep learning model designed to harness the power of AI, providing innovative and creative tools for designers. Using our algorithms, images of empty rooms can be transformed into fully furnished spaces based on text descriptions. Please refer to our <a href='https://github.com/Lavreniuk/generative-interior-design'>GitHub</a> for more details.</p>
    """
    with gr.Blocks() as demo:
        gr.Markdown(title)
        create_demo(model)

    demo.queue().launch(share=True)


if __name__ == '__main__':
    # Initialize pipelines

    
    main()
