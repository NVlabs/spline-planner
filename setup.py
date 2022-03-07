from setuptools import setup, find_packages

# read the contents of your README file
from os import path

this_directory = path.abspath(path.dirname(__file__))
with open(path.join(this_directory, "README.md"), encoding="utf-8") as f:
    lines = f.readlines()

# remove images from README
lines = [x for x in lines if ".png" not in x]
long_description = "".join(lines)

setup(
    name="Pplan",
    packages=[package for package in find_packages() if package.startswith("Pplan")],
    install_requires=[
        "numpy>=1.21.0",
        "jax",
        "pytorch-lightning",
        "wandb",
        "torch==1.10.2",
        "torchaudio==0.10.2",
        "torchvision==0.11.3",
        "torchtext",  # weird pytorch-lightning dependency bug
    ],
    eager_resources=["*"],
    include_package_data=True,
    python_requires=">=3",
    description="Policy planning",
    author="NVIDIA AV Research",
    author_email="yuxiaoc@nvidia.com",
    version="0.0.1",
    long_description=long_description,
    long_description_content_type="text/markdown",
)
