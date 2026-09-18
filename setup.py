from setuptools import find_packages, setup


setup(
    name="ne01-wholebody-v4-retargeting",
    version="0.4.0",
    description="Standalone NE01 WholeBody Omni V4 motion retargeting",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "mink", "mujoco", "numpy", "scipy", "qpsolvers[proxqp]", "daqp",
        "tqdm", "smplx", "torch", "imageio[ffmpeg]", "trimesh>=4.0",
        "usd-core>=24.0", "coacd>=1.0.0",
    ],
)
