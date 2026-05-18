"""
torch-sla: PyTorch Sparse Linear Algebra

A differentiable sparse linear equation solver library with multiple backends.
"""

import os
import sys
import subprocess
from setuptools import setup, find_packages
from setuptools.command.build_ext import build_ext

# Version
VERSION = "0.2.0"


def get_long_description():
    """Read README for long description"""
    readme_path = os.path.join(os.path.dirname(__file__), 'README.md')
    if os.path.exists(readme_path):
        with open(readme_path, 'r', encoding='utf-8') as f:
            return f.read()
    return ""


def check_cuda_available():
    """Check if CUDA toolkit is available"""
    try:
        result = subprocess.run(['nvcc', '--version'], 
                              capture_output=True, text=True)
        return result.returncode == 0
    except FileNotFoundError:
        return False


class CustomBuildExt(build_ext):
    """Custom build extension that handles CUDA compilation"""
    
    def build_extensions(self):
        # Check if we have CUDA
        if check_cuda_available():
            print("CUDA detected, building with CUDA extensions")
        else:
            print("CUDA not detected, only CPU extensions will be available")
        
        super().build_extensions()


# Extension modules (built at runtime via JIT for flexibility)
# For production, you may want to pre-compile these

setup(
    name='torch-sla',
    version=VERSION,
    author='Mingyuan Chi, Shizheng Wen',
    author_email='walker.chi.000@gmail.com, shizheng.wen@sam.math.ethz.ch',
    description='PyTorch Sparse Linear Algebra - Differentiable sparse solvers with CUDA support',
    long_description=get_long_description(),
    long_description_content_type='text/markdown',
    url='https://github.com/walkerchi/torch-sla',
    project_urls={
        'Bug Tracker': 'https://github.com/walkerchi/torch-sla/issues',
        'Documentation': 'https://github.com/walkerchi/torch-sla#readme',
        'Source': 'https://github.com/walkerchi/torch-sla',
    },
    
    packages=find_packages(),
    package_data={
        'torch_sla': [
            '../csrc/spsolve/*.cpp',
        ],
    },
    include_package_data=True,
    
    python_requires='>=3.8',
    install_requires=[
        'torch>=1.10.0',
        'ninja',  # For faster JIT compilation
    ],
    extras_require={
        'test': [
            'pytest>=6.0',
            'numpy>=1.19.0',
            'scipy>=1.5.0',
        ],
        'dev': [
            'pytest>=6.0',
            'numpy>=1.19.0',
            'scipy>=1.5.0',
            'black',
            'isort',
            'mypy',
        ],
        'docs': [
            'sphinx>=4.0',
            'furo',
            'sphinx-autodoc-typehints',
        ],
    },
    
    cmdclass={
        'build_ext': CustomBuildExt,
    },
    
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: Apache Software License',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Programming Language :: Python :: 3.12',
        'Programming Language :: C++',
        'Programming Language :: CUDA',
        'Topic :: Scientific/Engineering',
        'Topic :: Scientific/Engineering :: Mathematics',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ],
    
    keywords=[
        'pytorch', 'sparse', 'linear-algebra', 'cuda', 'cupy', 'cudss',
        'sparse-matrix', 'linear-solver', 'differentiable', 'autograd',
    ],
    
    zip_safe=False,
)
