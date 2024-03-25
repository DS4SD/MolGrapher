import setuptools
import platform

def install_torch(package: str, version: str = '', cpu: bool=True):
    """
    Creates path to download PyTorch (example: torch @ https://download.pytorch.org/whl/cpu/torch-2.1.2%2Bcpu-cp311-cp311-linux_x86_64.whl).
    Packages can be found at: https://download.pytorch.org/whl/.
    """
    cuda = "arm" not in platform.platform()
    if cpu:
        python_version = ''.join(platform.python_version().split('.')[:2])
        return ''.join([
            f'{package} @ https://download.pytorch.org/whl/',
            f'cpu/',
            f'{package}',
            f'-{version}' if version else '',
            '%2Bcpu',
            f'-cp{python_version}-cp{python_version}',
            'm' if int(python_version) <= 37 else '',
            '-linux_x86_64.whl',
        ])
    else:
        python_version = "311"
        cuda_version = "117"
        return ''.join([
            f'{package} @ https://download.pytorch.org/whl/',
            f'cu{cuda_version}/' if cuda else '',
            f'{package}',
            f'-{version}' if version else '',
            f'%2Bcu{cuda_version}' if cuda else '',
            f'-cp{python_version}-cp{python_version}',
            'm' if int(python_version) <= 37 else '',
            '-linux_x86_64.whl',
        ])
    
def install_paddle(package: str, version: str = ''):
    """
    Creates path to download PaddleOCR (example: paddlepaddle-gpu @ https://paddle-wheel.bj.bcebos.com/2.6.0/linux/linux-gpu-cuda11.7-cudnn8.4.1-mkl-gcc8.2-avx/paddlepaddle_gpu-2.6.0.post117-cp311-cp311-linux_x86_64.whl).
    Packages can be found at: https://paddle-wheel.bj.bcebos.com/.
    """
    python_version = "311"
    cuda_version = "117"
    cudnn_version = "8.4.1"
    return ''.join([
        f'{package} @ https://paddle-wheel.bj.bcebos.com/',
        f'{version}/linux/',
        f'linux-gpu-cuda{cuda_version}-cudnn{cudnn_version}-mkl-gcc8.2-avx/',
        f'paddlepaddle_gpu-{package}.post{cuda_version}-cp{python_version}-cp{python_version}-linux_x86_64.whl'
       
    ])

with open("README.md", "r") as fh:
    long_description = fh.read()

print(install_torch('torch', '2.1.2', cpu=True))


setuptools.setup(
    name="molgrapher",
    version="1.0.0",
    author="Lucas Morin",
    author_email="lum@zurich.ibm.com",
    description="A Python library",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.ibm.com/LUM/graph-recognition/",
    packages=setuptools.find_packages(exclude=["tests.*", "tests"]),
    install_requires=[
        "mol-depict @ git+ssh://git@github.ibm.com/LUM/molecule-depictor.git",
        "pytorch-lightning==2.1.3",
        "torch_geometric==2.4.0",
        "scikit-learn",
        "seaborn",
        "timm",
        "mahotas",
        "more_itertools",
        "rdkit==2023.09.5",
        "CairoSVG",
        "SmilesPE",
        "python-Levenshtein",
        "nltk",
        "ipykernel",
        "ipython",
        "rouge-score",
        "albumentations",
        "paddleocr",
        "torchsummary",
        "weighted-levenshtein"
    ],
    classifiers=[
        "Development Status :: 1 - Planning",
        "Intended Audience :: Developers",
        "License :: Other/Proprietary License",
        "Natural Language :: English",
        "Operating System :: OS Independent",
        "Topic :: Database",
        "Programming Language :: Python :: 3",
    ],
    python_requires='>=3.9', 
    package_data={"": ["*.json"]},
    extras_require={
        "gpu": [
            install_torch('torch', '2.0.1', cpu=False),
            install_torch('torchvision', '0.15.2', cpu=False),
            "tensorboard",
            "opencv-python",
            install_paddle('paddlepaddle-gpu', '2.6.0')
        ],
         "cpu": [
            install_torch('torch', '2.1.2', cpu=True),
            install_torch('torchvision', '0.16.2', cpu=True),
            "paddlepaddle"
        ]
    },
)