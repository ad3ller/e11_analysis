from setuptools import setup

setup(name="e11",
      version="0.0.51",
      description="process data from experiments",
      url="",
      author="Adam Deller",
      author_email="a.deller@ucl.ac.uk",
      license="BSD",
      packages=["e11"],
      install_requires=[
          "scipy>=0.14","numpy>=1.10","pandas>=0.17", "pyarrow>=0.12", "h5py>=2.5", "tqdm>=3.1.4"
      ],
      include_package_data=False,
      zip_safe=False)
