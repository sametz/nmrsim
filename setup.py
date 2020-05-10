import setuptools

with open("README.rst", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="nmrsim",
    version="0.5.1",
    author="Geoffrey M. Sametz",
    author_email="sametz@udel.edu",
    description="A library for simulating nuclear magnetic resonance (NMR) spectra.",
    long_description=long_description,
    long_description_content_type="text/x-rst",
    url="https://github.com/sametz/nmrsim",
    packages=setuptools.find_packages(
        exclude=['docs', 'jupyter', 'tests']
    ),
    include_package_data=True,  # so MANIFEST is recognized
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Education",
        "Programming Language :: Python :: 3 :: Only",
        "Topic :: Education",
        "Topic :: Scientific/Engineering :: Chemistry",
        "Topic :: Scientific/Engineering :: Physics",
        "Topic :: Scientific/Engineering :: Visualization",
    ],
    keywords='NMR simulation spectra spectrum',
    python_requires='>=3.6',
    install_requires=['matplotlib',
                      'numpy',
                      'sparse',
                      "importlib_resources ; python_version<'3.7'"
                      ],
    extras_require={
        'dev': [
            'flake8',
            'ipykernel',
            'jupyter',
            'nbsphinx',
            'pytest',
            'pyfakefs',
            'sphinx',
            'sphinx_rtd_theme',
            'sphinxcontrib-napoleon',
            # below are for current "extra" jupyter notebook features,
            # which may change as other dataviz options tested.
            'bokeh',
            'tox',
            'tox-pyenv',
        ]
    }
)
