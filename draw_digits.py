import torch
import torch.nn as nn
import numpy as np
from PIL import Image, ImageDraw
import tkinter as tk
from tkinter import messagebox

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

# Canvas for drawing
canvas = tk.Canvas(root, width=280, height=280, bg='white')
canvas.pack()

# PIL image for prediction
image = Image.new('L', (280, 280), 255)  # grayscale, white background
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
        canvas.create_line(last_x, last_y, event.x, event.y, width=18, fill='black', capstyle=tk.ROUND)
        draw.line([last_x, last_y, event.x, event.y], fill=0, width=18)  # black on white
        last_x, last_y = event.x, event.y

def on_mouse_up(event):
    global drawing
    drawing = False

canvas.bind("<Button-1>", on_mouse_down)
canvas.bind("<B1-Motion>", on_mouse_move)
canvas.bind("<ButtonRelease-1>", on_mouse_up)

# Label for prediction
prediction_label = tk.Label(root, text="Draw a digit and click Predict")
prediction_label.pack()

def clear_canvas():
    canvas.delete('all')
    global image, draw
    image = Image.new('L', (280, 280), 255)
    draw = ImageDraw.Draw(image)
    prediction_label.config(text="Draw a digit and click Predict")

def predict_digit():
    # Resize to 28x28
    resized = image.resize((28, 28))
    arr = np.array(resized)
    
    # Invert (MNIST has white digit on black)
    arr = 255 - arr
    arr = arr / 255.0
    arr = (arr - 0.5) / 0.5  # normalize
    
    # Predict
    input_tensor = torch.tensor(arr, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(input_tensor)
        pred = logits.argmax(dim=1).item()
        probs = torch.softmax(logits, dim=1)[0]
    
    prediction_label.config(text=f"Prediction: {pred} (Confidence: {probs[pred].item()*100:.1f}%)")

# Buttons
clear_btn = tk.Button(root, text="Clear", command=clear_canvas)
predict_btn = tk.Button(root, text="Predict", command=predict_digit)
clear_btn.pack(side=tk.LEFT)
predict_btn.pack(side=tk.RIGHT)

root.mainloop()