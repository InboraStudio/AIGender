
# AI Gender

```A Project Build to Test my New GPU 😊 ```

**AI Gender** is a compact, high-performance gender classification project that uses deep learning techniques to detect gender from facial images. This solution is engineered specifically for environments with **limited computational resources** such as:

* Legacy laptops and desktops
* Raspberry Pi devices
* Embedded systems
* Low-cost cloud VMs

The project is available as a **Gradio demo on Hugging Face Spaces**, and can be deployed or integrated into lightweight systems easily.
## About
The core of **AI Gender** is a **pre-trained convolutional neural network** model. It accepts an image input, processes it to extract facial features, and classifies the gender as either male or female.

Unlike large-scale models that require dedicated GPUs, this project prioritizes mathematical efficiency and architectural optimization.
## Mathematical Foundation

* Uses **Convolutional Neural Networks (CNNs)** optimized for small-scale inputs
* Reduces floating point operations (FLOPs) using smaller kernels and fewer channels
* Implements **transfer learning** with frozen backbone layers to minimize active computation
* Model size is trimmed using **quantization-aware training** (QAT)
* Confidence score output is based on softmax probability distribution

Key equations:

1. Convolution Operation:
   $(I * K)(x, y) = \sum_m \sum_n I(m, n) \cdot K(x - m, y - n)$

2. Softmax:
    $\text{Softmax}(z_i) = \frac{e^{z_i}}{\sum_j e^{z_j}}$

These optimizations ensure the model runs efficiently on minimal hardware.

---

## Low-End Technology Support

To make this model runnable on **super low-end tech**, we applied the following:

* **Backbone**: MobileNetV2 / ResNet18 (for small size and speed)
* **RAM Usage**: < 200MB during inference
* **Disk Footprint**: Model file under 10MB
* **CPU Only**: No CUDA or GPU dependencies
* **Lazy Loading**: Loads only necessary components on start
* **Zero External Heavy Libraries**: Avoids OpenCV, dlib, or other memory-intensive tools

This makes it suitable for real-time or batch inference on constrained devices.

---

## Demo (Hugging Face)

Try the model online here:

[https://huggingface.co/spaces/your-username/ai-gender](https://huggingface.co/spaces/DrChamyoung/AIGender)

---

## Notes

This project was created as an accessible AI showcase, proving that **practical deep learning** can be done without expensive hardware. It’s ideal for classrooms, research demonstrations, and offline AI use cases.
