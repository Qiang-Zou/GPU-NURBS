# NURBS_Inversion_GPU

- This project is written by Ziheng Bao and maintained by Yaonaiming Zhao, under the supervision of Qiang Zou
- email: qzou.code@gmail.com
- webpage: https://qiang-zou.github.io/
- Latest Release: 2024.06.14

## !important

The source code was developed as a byproduct of the projects and methods presented in [1].

It can be compiled with GCC 11.4.0+CUDA12.2 , and run on the operating system Ubuntu 22.04 LTS. Windows, Mac, and other compilers should also work.


1.Copyright
-----------

- NURBS_Inversion_GPU is GNU licensed. It is developed by  Ziheng Bao and maintained by Yaonaiming Zhao, under the supervision of Qiang Zou. All rights about the program are reserved by Ziheng Bao and Qiang Zou. This C++ source code is available only to a primary user for academic purposes. No secondary use, such as copy, distribution, diversion, business purpose, etc., is allowed. In no event shall the author be liable to any party for direct, indirect, special, incidental, or consequential damage arising out of the use of this program. NURBS_Inversion_GPU is self-contained.


2.Download
----------

- The source code can be downloaded from: 


3.Installing & Compiling (Ubuntu+GCC 11.4.0+CUDA12.2)
-------------------------------------------

- Make sure **CUDA** has been installed.
- Type <code>nvcc -V</code>  in your shell to check CUDA installation.
- Compile the project by the command <code>sh dev_compile.sh</code>.
- Get the executable file <code>test.o</code>.

## 4.Usage

After the compilation you can run the executable file <code>test.o</code>.

5.References
-------------

- [1] Ziheng Bao, Wu Liu, and Qiang Zou. Parallel Inverse Evaluation of NURBS Surfaces Based on Matrix Representation. Journal of Computer-Aided Design & Computer Graphics (2023):35.
