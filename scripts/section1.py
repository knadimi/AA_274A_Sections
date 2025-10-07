#!/usr/bin/env python3
import numpy as np
if __name__ == "__main__":
    # code goes here
    np.random.seed(42)
    x = np.random.normal(size=(4, 10))

# broadcasting approach (no loop)
D = np.sum((x[:, None, :] - x[None, :, :])**2, axis=2)
print(D)