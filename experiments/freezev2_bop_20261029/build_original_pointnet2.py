"""Build untouched GeDi PointNet2 kernels for the actual GPU architecture.

The upstream setup hardcodes obsolete GPU targets rejected by CUDA12. This
independent build entry point changes only target architecture and build paths;
the original Python/C++/CUDA dependency files remain untouched.
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
from pathlib import Path
import re
import sys


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gedi-source', required=True)
    parser.add_argument('--cuda-home', required=True)
    parser.add_argument('--wheel-directory', required=True)
    parser.add_argument('--architecture', default='8.6')
    args = parser.parse_args()
    source = Path(args.gedi_source).resolve() / 'backbones/pointnet2_ops_lib'
    cuda = Path(args.cuda_home).resolve()
    wheel_directory = Path(args.wheel_directory).resolve()
    wheel_directory.mkdir(parents=True, exist_ok=True)
    os.environ['CUDA_HOME'] = str(cuda)
    os.environ['PATH'] = str(Path(sys.executable).parent) + os.pathsep + str(cuda / 'bin') + os.pathsep + os.environ['PATH']
    os.environ['TORCH_CUDA_ARCH_LIST'] = args.architecture
    os.environ.setdefault('MAX_JOBS', '2')
    from setuptools import find_packages, setup
    from torch.utils.cpp_extension import BuildExtension, CUDAExtension
    import torch
    # PyTorch's ATen CUDA headers include cuBLAS/cuSPARSE/cuSOLVER interfaces.
    # Reuse the development headers shipped in its already-installed wheels.
    nvidia_headers = [str(path) for path in (Path(torch.__file__).parent.parent / 'nvidia').glob('*/include') if path.is_dir()]

    os.chdir(source)
    files = glob.glob('pointnet2_ops/_ext-src/src/*.cpp') + glob.glob('pointnet2_ops/_ext-src/src/*.cu')
    records = {}
    for path in sorted(Path('pointnet2_ops').rglob('*')):
        if path.is_file() and path.suffix in {'.py', '.cpp', '.cu', '.h'}:
            records[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    version = re.search(r'__version__\s*=\s*[\'"]([^\'"]+)', Path('pointnet2_ops/_version.py').read_text()).group(1)
    (wheel_directory / 'pointnet2_build_sources.json').write_text(json.dumps({
        'source_root': str(source), 'files_sha256': records, 'architecture': args.architecture,
        'cuda_home': str(cuda), 'torch': torch.__version__, 'cuda': torch.version.cuda,
        'scientific_source_files_modified': False, 'upstream_setup_architecture_override': True,
    }, indent=2) + '\n')
    sys.argv = [sys.argv[0], 'bdist_wheel', '--dist-dir', str(wheel_directory)]
    setup(
        name='pointnet2_ops', version=version, packages=find_packages(),
        ext_modules=[CUDAExtension(
            name='pointnet2_ops._ext', sources=files,
            extra_compile_args={'cxx': ['-O3'], 'nvcc': ['-O3', '-Xfatbin', '-compress-all']},
            include_dirs=[str(source / 'pointnet2_ops/_ext-src/include'), *nvidia_headers],
        )],
        cmdclass={'build_ext': BuildExtension}, include_package_data=True,
    )
