"""Hypothesis: py36 importlib_resources is incompatible with pytest/pyfakefs.
Creating minimal case for testing path-mangling.
"""
import sys
if sys.version_info >= (3, 7):
    from importlib import resources
else:
    import importlib_resources as resources

import nmrsim.bin


def findbin():
    init_path_context = resources.path(nmrsim.bin, '__init__.py')
    with init_path_context as p:
        init_path = p
    bin_path = init_path.parent
    return bin_path
