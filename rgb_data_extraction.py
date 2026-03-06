import numpy as np
import rasterio
import numpy as np

tif_path = "./UH_NAD83_271460_3289689.tif"
with rasterio.open(tif_path) as ds:
    rgb = ds.read([1, 2, 3])
    rgb = np.moveaxis(rgb, 0, -1)

import matplotlib.pyplot as plt

print(rgb.shape)
plt.figure()
plt.imshow(rgb)
plt.show()
