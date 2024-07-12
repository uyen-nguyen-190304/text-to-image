import os
from flask import Flask, request, render_template
from text2img_model import create_pipeline, text2img
import torch

app = Flask(__name__)

# Ensure the static directory exists
if not os.path.exists('static'):
    os.makedirs('static')

pipeline = create_pipeline(model_name="dreamlike-art/dreamlike-photoreal-2.0")

@app.route("/", methods=['GET', 'POST'])
def index():
    if request.method == "GET":
        return render_template("index.html")
    else:
        user_input = request.form["prompt"]
        num_images = int(request.form["num_images"])  # Convert to integer

        print("Start generating...")
        torch.cuda.empty_cache()  # Clear the CUDA cache

        images = text2img(user_input, num_images, pipeline)  # Generate images
        print("Finish generating...")

        # Save images and create URLs for them
        image_urls = []
        for i, img in enumerate(images):
            img_path = f"static/output_{i}.jpg"
            img.save(img_path)
            image_urls.append(img_path)

        return render_template("index.html", image_urls=image_urls, prompt=user_input)

if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=8888, use_reloader=False)