import numpy as np
import io

# Specify the path to the binary file containing the NumPy data.
file_path = "flwr_app/output"

with open(file_path, "rb") as f:
    binary_data = f.read()

# Try loading NumPy arrays directly
try:
    tensor = np.load(io.BytesIO(binary_data), allow_pickle=True)
    print("Extracted tensor shape:", tensor.shape)
except Exception as e:
    print("Error extracting tensor:", e)
