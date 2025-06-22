import setuptools
from setuptools import setup

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="meshoid",
    version="1.47.1",
    description="Package for analysis of meshless simulation data",
    #      long_description=long_description,
    url="http://github.com/mikegrudic/meshoid",
    author="Mike Grudić",
    author_email="mike.grudich@gmail.com",
    license="MIT",
    project_urls={
        "Bug Tracker": "https://github.com/mikegrudic/meshoid",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    package_dir={"": "src"},
    packages=["meshoid", "meshoid.radiation"],
    python_requires=">=3.6",
    zip_safe=False,
    include_package_data=True,
)
