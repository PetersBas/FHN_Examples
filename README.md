# FHN_Examples
Examples of applications of fully hyperbolic invertible neural networks, corresponding to the paper **Fully invertible hyperbolic neural networks for large-scale surface and sub-surface characterization via remote sensing** (submitted). The examples show several applications where fully invertible networks do not directly apply. The modifications/specializations in the paper enable the application to the a wider variety of problems, while not giving up full-invertibility.


This repo contains the driver scripts for the examples:

- **Semantic segmentation/interpolation of 3D seismic imagery**. This script is pretty much self contained for educational purposes. Uses 248x248X248x12 input blocks, 6144 channels, and 30 network layers. The fully invertible hyperbolic network reduces the memory requirements for the states from 21.96GB to 2.19 GB. The block low-rank layer reduces the memory requirments for the convolutional kernels from 41.16 GB to 0.23 GB.
- **Time-lapse hyperspectral land-use change detection**. The fully invertible hyperbolic network reduces the memory requirements for the states from 22.5 GB to 3.7 GB. The input data to the network are blocks of size 368 × 288 × 184 x 16. This application shows how to map a 4D time-lapse hyperspectral dataset into a 2D change map, while also reducing the output resolution, and not giving up invertibility.
- **Aquifer mapping**: not available at the moment because the data is not open-source yet.

Examples are tested in Julia 1.9.
