from setuptools import setup, find_packages


def get_requirements():
    with open("requirements.txt") as f:
        return f.read().splitlines()


setup(
    name="Chan-Vese",
    version="1.0.0",
    packages=find_packages(),
    author_email="r.rodriguezr.2020@alumnos.urjc.es",
    description="Chan-Vese segmentation providing an interface including multiphase and multichannel extensions",
    url="https://github.com/Ruben-Rodriguez-Redondo/TFG-Math-Chan-Vese",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.10, <3.11",
    install_requires=get_requirements(),
)
