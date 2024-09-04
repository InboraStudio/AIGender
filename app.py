import gradio as gr
from fastai.vision.all import *
import pathlib

# Adjust the path handling for compatibility between different OS
temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath
pathlib.PosixPath = temp

# Load your pre-trained model
learn = load_learner('model.pkl')
labels = learn.dls.vocab

# Prediction function
def predict(img):
   img = PILImage.create(img)
   pred, pred_idx, probs = learn.predict(img)
   return {labels[i]: float(probs[i]) for i in range(len(labels))}

# Title and description
title = "Female/Male Classifier"
description = "A Female/Male classifier trained By Dr chamyoung."
# Ensure the correct path is given to your example images
examples = ['femaleDefault.jpg', 'maleDefault.jpg']

# Update the Gradio Interface
inter = gr.Interface(
   fn=predict,
   inputs=gr.Image(type="pil"),
   outputs=gr.Label(),
   title=title,
   description=description,
   examples=examples,  # Make sure the images are correctly referenced
   cache_examples=False,  # Disable example caching
   examples_per_page=2
)

inter.queue()
inter.launch()
