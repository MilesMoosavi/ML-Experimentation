import torch
import torch.nn as nn
import numpy as np
from PIL import Image, ImageDraw, ImageTk
import tkinter as tk
from tkinter import messagebox
import os
import random
from skimage.metrics import mean_squared_error
from skimage.metrics import structural_similarity as ssim

# Define the Neural Network Model (same as in notebook)
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(28*28, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Load the model
model = SimpleNN()
model.load_state_dict(torch.load('mnist_model.pth', map_location='cpu'))
model.eval()

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

# Tkinter GUI
root = tk.Tk()
root.title("MNIST Digit Drawer")

# Main frame
main_frame = tk.Frame(root)
main_frame.pack()

# Left frame for canvas and controls
left_frame = tk.Frame(main_frame)
left_frame.pack(side=tk.LEFT)

# Right frame for sample digits grid
right_frame = tk.Frame(main_frame, width=280, height=280)
right_frame.pack(side=tk.RIGHT)

# Canvas for drawing
canvas = tk.Canvas(left_frame, width=280, height=280, bg="#000000")
canvas.pack()

# PIL image for prediction
image = Image.new('L', (280, 280), 0)  # pure black background
draw = ImageDraw.Draw(image)

# Drawing state
drawing = False
last_x, last_y = None, None

def on_mouse_down(event):
    global drawing, last_x, last_y
    drawing = True
    last_x, last_y = event.x, event.y

def on_mouse_move(event):
    global drawing, last_x, last_y
    if drawing and last_x is not None and last_y is not None:
        canvas.create_line(last_x, last_y, event.x, event.y, width=18, fill='white', capstyle=tk.ROUND)
        draw.line([last_x, last_y, event.x, event.y], fill=255, width=18)  # white on gray
        last_x, last_y = event.x, event.y

def on_mouse_up(event):
    global drawing
    drawing = False

canvas.bind("<Button-1>", on_mouse_down)
canvas.bind("<B1-Motion>", on_mouse_move)
canvas.bind("<ButtonRelease-1>", on_mouse_up)

# Ensure functions are defined before button creation
def clear_canvas():
    canvas.pack()  # Show the canvas
    compressed_image_label.pack_forget()  # Hide the compressed image
    canvas.delete('all')
    global image, draw, compressed_image_ref
    image = Image.new('L', (280, 280), 0)  # pure black background
    draw = ImageDraw.Draw(image)
    prediction_label.config(text="Draw a digit and click Predict")
    compressed_image_ref = None  # Clear the reference
    # Clear the right frame
    for widget in right_frame.winfo_children():
        widget.destroy()

# Update display_sample_digits to show similarity for all 4 results and use percentage similarity
def compute_similarity(image1, image2):
    # Resize both images to 28x28
    image1_resized = image1.resize((28, 28))
    image2_resized = image2.resize((28, 28))
    
    # Convert to numpy arrays
    arr1 = np.array(image1_resized, dtype=np.float32)
    arr2 = np.array(image2_resized, dtype=np.float32)
    
    # Normalize pixel values
    arr1 /= 255.0
    arr2 /= 255.0
    
    # Compute Structural Similarity Index (SSIM)
    similarity = ssim(arr1, arr2, data_range=1.0)  # Specify data_range explicitly
    return similarity * 100  # Convert to percentage

def display_sample_digits(pred):
    # Clear right frame
    for widget in right_frame.winfo_children():
        widget.destroy()
    
    sample_dir = f'mnist_samples/digit_{pred}/'
    if not os.path.exists(sample_dir):
        return  # No samples
    
    files = [f for f in os.listdir(sample_dir) if f.endswith('.png')]
    if len(files) < 4:
        return  # Not enough samples
    
    # Compute similarity scores
    similarities = []
    for file in files:
        img_path = os.path.join(sample_dir, file)
        sample_img = Image.open(img_path)
        similarity = compute_similarity(image.resize((28, 28)), sample_img)
        similarities.append((similarity, file))
    
    # Sort by similarity (descending order)
    similarities.sort(reverse=True, key=lambda x: x[0])
    
    # Select top 4
    top_samples = similarities[:4]
    
    image_refs = []  # Store references to images
    
    for i, (similarity, file) in enumerate(top_samples):
        img_path = os.path.join(sample_dir, file)
        img = Image.open(img_path)
        img = img.resize((140, 140))
        photo = ImageTk.PhotoImage(img)  # Use ImageTk.PhotoImage
        image_refs.append(photo)  # Keep reference in list
        
        label = tk.Label(right_frame, image=photo)
        label.grid(row=i // 2 * 2, column=i % 2)  # Ensure proper placement
        
        # Add similarity score below the image
        similarity_label = tk.Label(right_frame, text=f"Similarity: {similarity:.2f}%")
        similarity_label.grid(row=i // 2 * 2 + 1, column=i % 2)  # Place directly below the image

    # Attach image_refs to a global variable to prevent garbage collection
    global image_refs_global
    image_refs_global = image_refs

def predict_digit():
    global compressed_image_ref
    # Resize to 28x28
    resized = image.resize((28, 28))
    
    arr = np.array(resized)
    
    # Adjust for black background and white digit
    arr = np.clip(arr, 0, 255)  # Ensure values are within valid range
    arr = arr / 255.0
    arr = (arr - 0.5) / 0.5  # normalize
    
    # Predict
    input_tensor = torch.tensor(arr, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(input_tensor)
        pred = logits.argmax(dim=1).item()
        probs = torch.softmax(logits, dim=1)[0]
    
    prediction_label.config(text=f"Prediction: {pred} (Confidence: {probs[pred].item()*100:.1f}%)")
    
    # Display the compressed image in place of the canvas
    compressed_image_ref = ImageTk.PhotoImage(resized.resize((280, 280)))
    compressed_image_label.config(image=compressed_image_ref)
    
    canvas.pack_forget()  # Hide the canvas
    compressed_image_label.pack()  # Show the compressed image
    
    # Display sample digits
    display_sample_digits(pred)

# Fix: Move button creation after function definitions
button_frame = tk.Frame(left_frame)
button_frame.pack(side=tk.BOTTOM, fill=tk.X)

clear_btn = tk.Button(button_frame, text="Clear", command=clear_canvas)
predict_btn = tk.Button(button_frame, text="Predict", command=predict_digit)
clear_btn.pack(side=tk.LEFT)
predict_btn.pack(side=tk.RIGHT)

# Move prediction label to its own frame above the buttons
label_frame = tk.Frame(left_frame)
label_frame.pack(side=tk.BOTTOM, fill=tk.X)

prediction_label = tk.Label(label_frame, text="Draw a digit and click Predict")
prediction_label.pack()

# Add a label to display the compressed image
compressed_image_label = tk.Label(left_frame)
compressed_image_label.pack()

# Fix: Use a global variable to store the compressed image reference
compressed_image_ref = None  # Global variable to store the compressed image

root.mainloop()