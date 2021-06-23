# Notes on Testing `sparse` Problems

Starting with sparse 0.11, `nmrsim` code broke. Tracking testing notes here.

My original nmrsim conda environment used:

- python 3.7.5
- sparse 0.8.0
- numpy 1.17.4
- scipy 1.3.2
- numba 0.46.0

And tests pass. 

In project binder environments, builds were failing. 
It was found that Sparse versions >0.10 were failing at the tensordot function 
in qm.hamiltonian_sparse().

sparse 0.11 rewrote tensordot. 
0.12 fixed some reported errors that seem adjacent to my errors, 
but 0.12 does not fix nmrsim.

On my PC, I created a virtualenv with:

- python 3.9.5
- sparse 0.12.0
- numpy 1.20.3
- scipy 1.6.3
- numba 0.53.1

and the same broken tests were seen.

I created a conda python 3.9 environment on my mac with:

- python 3.9.5
- sparse 0.12.0
- numpy 1.20.3
- scipy 1.7.0
- numba 0.53.1

and it behaves similarly.

I copied a "traditional" tensordot example from 
[the numpy docs](https://numpy.org/doc/stable/reference/generated/numpy.tensordot.html?highlight=tensordot#numpy.tensordot),
made a test for it,
and made a version of that for sparse.tensordot 
(which is supposed to be a direct substitution).
the numpy test passes but the sparse test produces an error 
from the sparse.tensordot function:

```
  at = a.transpose(newaxes_a).reshape(newshape_a)
        bt = b.transpose(newaxes_b).reshape(newshape_b)
        res = _dot(at, bt, return_type)
>       return res.reshape(olda + oldb)
E       AttributeError: 'NoneType' object has no attribute 'reshape'

../../../../anaconda3/anaconda3/envs/nmrsim_39/lib/python3.9/site-packages/sparse/_common.py:146: AttributeError
```