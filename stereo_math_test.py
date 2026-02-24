import numpy as np

params = np.load("stereo_params.npz")
f = params["fx"].item()
B = params["baseline_cm"].item()

print("Loaded f:", f, "B:", B)

# Example: test with some disparity
d = 60  # pixels (example)
Z = (f * B) / d
print("Estimated Z:", Z, "cm")