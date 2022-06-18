# Simple-FDTD-based-on-taichi
This repository includes some python code based on taichi parallel computation package to simulate real-time electric-magnetic field and rendering it.
## FDTD
*FDTD*, Finite difference time domain method, is a tool for simulation of electric-magnetic field dynamically, which is usually running on CPU and professional applications like ansys in past. With taichi tool package, the process of field responsing to electric mass move or simple harmonic pulse source can be visualized.
## Taichi tool
In this project, taichi tools transfer heavy calculation work from CPU to GPU and render the window so that the real-time simulation visualization becomes possible. In order to realise cross-platform function, taichi kernel is initialized with *arch = ti.vulkan* and the original platform is a X86 Macbook pro.
