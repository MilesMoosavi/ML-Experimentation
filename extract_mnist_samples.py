import os
import random
from torchvision import datasets

# Path to MNIST data
MNIST_PATH = "data/MNIST"
OUTPUT_DIR = "mnist_samples"
COUNT = 100

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Ensure MNIST files are in the correct location
raw_dir = os.path.join(MNIST_PATH, "raw")
processed_dir = os.path.join(MNIST_PATH, "processed")

if not os.path.exists(processed_dir):
    print("Processing raw MNIST files...")
    os.makedirs(processed_dir, exist_ok=True)
    # Add logic to process raw files into the expected format if needed
else:
    print("MNIST files detected and ready.")

# Load MNIST test set
mnist_test = datasets.MNIST(root=MNIST_PATH, train=False, download=True)

# Extract 100 samples for each digit (0-9)
for digit in range(10):
    indices = [i for i, label in enumerate(mnist_test.targets) if label.item() == digit]
    chosen = random.sample(indices, min(COUNT, len(indices)))
    digit_dir = os.path.join(OUTPUT_DIR, f"digit_{digit}")
    os.makedirs(digit_dir, exist_ok=True)
    for i, idx in enumerate(chosen):
        img, label = mnist_test[idx]
        img.save(os.path.join(digit_dir, f"mnist_{i:03d}_label_{label}.png"))
    print(f"Saved {len(chosen)} samples of digit {digit} to {digit_dir}/")
