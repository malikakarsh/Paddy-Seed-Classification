import cv2
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk

kernel_size = (11,11)

def apply_blur(image):
    blurred = cv2.GaussianBlur(image, kernel_size, 0)
    return blurred

def apply_canny(image, threshold):
    mini = min(int(threshold) * 2, 255)
    canny = cv2.Canny(image, int(threshold), int(mini))
    return canny

def update_image(threshold):
    canny = apply_canny(blurred, threshold)
    
    img = cv2.cvtColor(canny, cv2.COLOR_GRAY2RGB)
    
    pil_img = Image.fromarray(img)
    pil_img = pil_img.resize((600, 800))
    
    tk_img = ImageTk.PhotoImage(pil_img)
    image_label.config(image=tk_img)
    image_label.image = tk_img  

def open_image():
    file_path = filedialog.askopenfilename()
    
    global image, blurred
    image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
    blurred = apply_blur(image)
    
    img = cv2.cvtColor(blurred, cv2.COLOR_GRAY2RGB)
    
    pil_img = Image.fromarray(img)
    pil_img = pil_img.resize((600, 800))
    
    tk_img = ImageTk.PhotoImage(pil_img)
    image_label.config(image=tk_img)
    image_label.image = tk_img  
    
    initial_threshold = 0
    threshold_scale = tk.Scale(root, from_=0, to=255, orient=tk.HORIZONTAL,
                               label="Threshold",
                               command=update_image)
    threshold_scale.set(initial_threshold)
    threshold_scale.pack()

# root window
root = tk.Tk()
root.title("Edge Detector")
root.geometry("600x900")

image_label = tk.Label(root)
image_label.pack(fill=tk.BOTH, expand=True)

open_button = tk.Button(root, text="Open Image", command=open_image)
open_button.pack()

# main event loop
root.mainloop()
