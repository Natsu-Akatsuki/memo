from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='test_extension',
    ext_modules=[
        CUDAExtension(
            name='test_extension',
            sources=['test_extension.cu'],
            extra_compile_args={'cxx': ['-g'],
                                'nvcc': ['-O3', '-Wno-deprecated-gpu-targets']})
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)
