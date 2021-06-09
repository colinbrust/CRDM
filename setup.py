import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="DroughtCast",
    version="0.1",
    author="Colin Brust",
    author_email="colin.brust@gmail.com",
    description="Machine Learning Forecast of the USDM",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/colinbrust/DroughtCAst",
    packages=setuptools.find_packages(),
    # classifiers=[
    #     "Programming Language :: Python :: 3",
    #     "License :: OSI Approved :: MIT License",
    #     "Operating System :: OS Independent",
    # ],
    python_requires='>=3.6',
)