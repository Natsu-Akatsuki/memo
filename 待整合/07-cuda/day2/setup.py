from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='hello_extension',
    ext_modules=[
        CUDAExtension('hello', [
            'hello_extension.cu',
        ])
    ] ,
    cmdclass={
        'build_ext': BuildExtension     
    }
)
