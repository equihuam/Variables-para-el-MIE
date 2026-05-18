import rasterio
import numpy as np

path = r"C:\wf-ie-data\results\reference\region_2\ref_grid.tif"

with rasterio.open(path) as src:
    arr = src.read(1)
    nodata = src.nodata
    total = arr.size
    if nodata is None:
        valid = np.isfinite(arr).sum()
    else:
        valid = np.sum((arr != nodata) & np.isfinite(arr))

print("total celdas:", total)
print("celdas válidas:", valid)