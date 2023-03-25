# Image Resizing

Script to resize the image based on the aspect ratio of the actual image using Lanczos filter. Lanczos filter uses a windowed sinc function to compute the weights for each pixel in the resized image, which results in high-quality resampling with minimal loss of detail. Lanczos filter is known for its ability to preserve image sharpness and is often preferred over other resampling filters like bilinear or bicubic.

To run:

Install the required dependencies:

```python
python3 -m pip install -r requirements.txt
```

```python
python3 script.py
```
