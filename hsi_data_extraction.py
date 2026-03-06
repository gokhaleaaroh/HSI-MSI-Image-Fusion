import spectral as sp
import numpy as np

pix_path = "cube.pix"
hdr_path = "cube.hdr"

img = sp.envi.open(hdr_path, pix_path)

cube = np.asarray(img.load()) 

print(cube.shape, cube.dtype)

import matplotlib.pyplot as plt

plt.figure()
plt.imshow(cube[:, :, 10])
plt.show()
