from pathlib import Path
import os
import glob

from setuptools import setup, find_packages

root_dir = Path(__file__).parent

assets_fns = glob.glob("sapien_env/assets/**", recursive=True)
assets_fns = [f for f in assets_fns if os.path.isfile(f)]
assets_fns = [os.path.join(*(f.split(os.path.sep)[1:])) for f in assets_fns]

setup(
    packages=find_packages(),
    python_requires=">=3.8",
    # install_requires=[
    #     "sapien",
    #     "natsort",
    #     "numpy",
    #     "transforms3d",
    #     "gym==0.25.2",
    #     "open3d>=0.15.2",
    #     "imageio",
    #     "torch>=1.11.0",
    #     "nlopt",
    #     "smplx",
    #     "opencv-python",
    #     "mediapipe",
    #     "torchvision",
    #     "record3d",
    #     "pyperclip",
    # ],
    extras_require={"tests": ["pytest", "black", "isort"]},
    package_data={'sapien_env': assets_fns,},
    include_package_data=True,
)
