# ImaginArt: AI-Powered Text to Image Generator

## Overview

The **ImaginArt: AI-Powered Text to Image Generator** is a web application that uses a Stable Diffusion model to generate images based on user-defined text prompts. This project provides a simple and interactive interface for users to create multiple images quickly and easily.

## Features

- **User-Friendly Interface**: A clean web interface that allows users to input prompts and select the number of images to generate.
- **Dynamic Image Generation**: Users can generate 1 to 15 images at once based on their prompt.
- **Responsive Design**: Images are displayed in a centered and visually appealing grid layout.

## Technologies Used

- **Flask**: A lightweight WSGI web application framework for Python, used for creating the web server.
- **PyTorch**: An open-source machine learning library that provides the backend for model training and inference.
- **Diffusers**: A library that includes various state-of-the-art diffusion models, including Stable Diffusion.
## Installation

### Prerequisites

- Python 3.8 or higher
- CUDA-compatible GPU (for faster inference) or CPU

### Setup Steps

1. Clone the repository:

   ```bash
   git clone https://github.com/uyen-nguyen-190304/text-to-image
   cd text-to-image
   ```

2. Set up a virtual environment:

   ```bash
   python -m venv myenv
   source myenv/bin/activate  # On Windows use `myenv\Scripts\activate`
   ```

3. Install the required packages:

   ```bash
   pip install -r requirements.txt
   ```

4. Run the Flask application:

   ```bash
   python svr_model.py
   ```

5. Open your web browser and navigate to `http://127.0.0.1:8888`.

## Usage

1. Select the number of images to generate from the dropdown menu.
2. Enter a prompt in the text area.
3. Click the "Generate" button to create images based on your prompt.
4. The generated images will be displayed below the prompt with the prompt text clearly labeled.

## Folder Structure

```
text-to-img/
│
├── static/                # Directory for storing generated images and static files
│   ├── output_0.jpg       # Example generated image
│   ├── output_1.jpg       # Example generated image
│   └── ...                # Additional generated images
│
├── templates/             # Directory for HTML templates
│   ├── index.html         # Main HTML template for the web interface
│
└── requirements.txt       # Text file with all requirements for the projetc
│
├── text2img_model.py      # Module for Stable Diffusion model and image generation logic
│
└── svr_model.py           # Flask application to serve the web interface
```

## **Contact**
For any questions or issues, please contact Uyen Nguyen via [nguyen_u1@denison.edu](mailto:nguyen_u1@denison.edu).