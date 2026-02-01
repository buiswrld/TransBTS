import pickle
import numpy as np
import os

os.makedirs("data/pkls", exist_ok=True)

for i in range(3):
    arr = np.random.randint(0, 255, (64, 64), dtype=np.uint8)
    with open(f"data/pkls/dummy_{i}.pkl", "wb") as f:
        pickle.dump(arr, f)
