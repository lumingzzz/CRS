#!/usr/bin/env python

from setuptools import setup

import os
import subprocess
import time
import torch
from torch.utils.cpp_extension import (BuildExtension, CppExtension,
                                       CUDAExtension)


def make_cuda_ext(name, module, sources, sources_cuda=None):
    if sources_cuda is None:
        sources_cuda = []
    define_macros = []
    extra_compile_args = {'cxx': []}

    if torch.cuda.is_available() or os.getenv('FORCE_CUDA', '0') == '1':
        define_macros += [('WITH_CUDA', None)]
        extension = CUDAExtension
        extra_compile_args['nvcc'] = [
            '-D__CUDA_NO_HALF_OPERATORS__',
            '-D__CUDA_NO_HALF_CONVERSIONS__',
            '-D__CUDA_NO_HALF2_OPERATORS__',
        ]
        sources += sources_cuda
    else:
        print(f'Compiling {name} without CUDA')
        extension = CppExtension

    return extension(
        name=f'{module}.{name}',
        sources=[os.path.join(*module.split('.'), p) for p in sources],
        define_macros=define_macros,
        extra_compile_args=extra_compile_args)


if __name__ == '__main__':
    ext_modules = [
        make_cuda_ext(
            name='deform_conv_ext',
            module='modules.dcn',
            sources=['src/deform_conv_ext.cpp'],
            sources_cuda=[
                'src/deform_conv_cuda.cpp',
                'src/deform_conv_cuda_kernel.cu'
            ]),
    ]

    setup(
        name='DCNv2',
        ext_modules=ext_modules,
        cmdclass={'build_ext': BuildExtension},
        zip_safe=False)
