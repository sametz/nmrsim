"""qm is an attempt to reorganize the API indirectly. Instead of moving all
the quantum mechanical calculations to this file, we try to import them instead.

If this works, we can leave functions in their original locations for now and
essentially mock the new API.
"""

from .nmrmath import hamiltonian, hamiltonian_slow

