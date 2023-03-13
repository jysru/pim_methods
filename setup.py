import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="pim-methods",
    version="0.1",
    author="Jysru",
    author_email="jysru@pm.me",
    description="PIM methods for array convergence to target/matrix retrieval of linear systems.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/jysru/pim_methods",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)