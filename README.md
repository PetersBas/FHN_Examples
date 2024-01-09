# FHN_Examples
Examples of applications of fully hyperbolic invertible neural networks, corresponding to the paper **Fully invertible hyperbolic neural networks for large-scale surface and sub-surface characterization via remote sensing** (submitted). The examples show several applications where fully invertible networks do not directly apply. The modifications/specializations in the paper enable the application to the a wider variety of problems, while not giving up full-invertibility.


This repo contains the driver scripts for the examples:

- **Semantic segmentation/interpolation of 3D seismic imagery**. This script is pretty much self contained for educational purposes. Uses 248x248X248x4 input blocks, 2048 channels, and 30 network layers. The fully invertible hyperbolic network reduces the memory requirements for the states from 175.71 GB to 5.86 GB. The block low-rank layer reduces the memory requirments for the convolutional kernels from 36.58 GB to 0.61 GB.
- **Time-lapse hyperspectral land-use change detection**. The fully invertible hyperbolic network reduces the memory requirements for the states from 306.61 GB to 17.03 GB. The input data to the network are blocks of size 304 × 240 × 152 x 8. This application shows how to map a 4D time-lapse hyperspectral dataset into a 2D change map, while also reducing the output resolution, and not giving up invertibility.
- **Aquifer mapping**: not available at the moment because the data is not open-source yet.

Examples are tested in Julia 1.9.
