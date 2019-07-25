import setuptools

with open("README.rst", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="nmrtools",
    version="0.2.0",
    author="Geoffrey M. Sametz",
    author_email="sametz@udel.edu",
    description="A library for simulating nuclear magnetic resonance (NMR) spectra.",
    long_description=long_description,
    long_description_content_type="text/x-rst",
    url="https://github.com/sametz/nmrsim",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    keywords='NMR simulation spectra spectrum',
    install_requires=['matplotlib',
                      'numpy',
                      'sparse'],
    extras_require={
        'dev': [
            'pytest',
            'pyfakefs',
        ]
    }
)