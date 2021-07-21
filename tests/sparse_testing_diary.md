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

I parametrized the numpy "traditional" test. 
If either a or b is COO, or both, the tests pass on all sparse versions. 
If they are both ndarray, the test fails on all sparse versions.

I parametrized the 2-spin v, Lz test, mixing ndarray and sparse types.
Under Sparse 0.10, only the double ndarray fails, as above.
Under Sparse 0.12, all fail:

- double ndarray gives the NoneType/reshape error as in the traditional test.
- one ndarray, one sparse gives error ending in:
```
../../../anaconda3/anaconda3/envs/nmrsim_39/lib/python3.9/site-packages/numba/core/dispatcher.py:420: in _compile_for_args
    error_rewrite(e, 'typing')
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 

e = TypingError('Failed in nopython mode pipeline (step: nopython frontend)\nNo implementation of function Function(<built...1:\n                    out[oidx1, oidx2] += data1[didx1] * array2[oidx2, coords1[1, didx1]]\n                    ^\n')
issue_type = 'typing'

    def error_rewrite(e, issue_type):
        """
        Rewrite and raise Exception `e` with help supplied based on the
        specified issue_type.
        """
        if config.SHOW_HELP:
            help_msg = errors.error_extras[issue_type]
            e.patch_message('\n'.join((str(e).rstrip(), help_msg)))
        if config.FULL_TRACEBACKS:
            raise e
        else:
>           raise e.with_traceback(None)
E           numba.core.errors.TypingError: Failed in nopython mode pipeline (step: nopython frontend)
E           No implementation of function Function(<built-in function setitem>) found for signature:
E            
E            >>> setitem(array(float64, 2d, C), UniTuple(int64 x 2), complex128)
E            
E           There are 16 candidate implementations:
E                - Of which 16 did not match due to:
E                Overload of function 'setitem': File: <numerous>: Line N/A.
E                  With argument(s): '(array(float64, 2d, C), UniTuple(int64 x 2), complex128)':
E                 No match.
E           
E           During: typing of setitem at /Users/geoffreysametz/anaconda3/anaconda3/envs/nmrsim_39/lib/python3.9/site-packages/sparse/_common.py (1026)
E           
E           File "../../../anaconda3/anaconda3/envs/nmrsim_39/lib/python3.9/site-packages/sparse/_common.py", line 1026:
E               def _dot_coo_ndarray(coords1, data1, array2, out_shape):  # pragma: no cover
E                   <source elided>
E                           while didx1 < len(data1) and coords1[0, didx1] == oidx1:
E                               out[oidx1, oidx2] += data1[didx1] * array2[oidx2, coords1[1, didx1]]
E                               ^

../../../anaconda3/anaconda3/envs/nmrsim_39/lib/python3.9/site-packages/numba/core/dispatcher.py:361: TypingError
```
- double sparse was giving the error I was seeing at the start of all this:
```
../../../anaconda3/anaconda3/envs/nmrsim_39/lib/python3.9/site-packages/numba/core/dispatcher.py:420: in _compile_for_args
    error_rewrite(e, 'typing')
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 

e = TypingError('Failed in nopython mode pipeline (step: nopython frontend)\nNo implementation of function Function(<built...oo_coo(\n        <source elided>\n                ):\n                    sums[k] += av * bv\n                    ^\n')
issue_type = 'typing'

    def error_rewrite(e, issue_type):
        """
        Rewrite and raise Exception `e` with help supplied based on the
        specified issue_type.
        """
        if config.SHOW_HELP:
            help_msg = errors.error_extras[issue_type]
            e.patch_message('\n'.join((str(e).rstrip(), help_msg)))
        if config.FULL_TRACEBACKS:
            raise e
        else:
>           raise e.with_traceback(None)
E           numba.core.errors.TypingError: Failed in nopython mode pipeline (step: nopython frontend)
E           No implementation of function Function(<built-in function setitem>) found for signature:
E            
E            >>> setitem(array(float64, 1d, C), int64, complex128)
E            
E           There are 16 candidate implementations:
E                 - Of which 16 did not match due to:
E                 Overload of function 'setitem': File: <numerous>: Line N/A.
E                   With argument(s): '(array(float64, 1d, C), int64, complex128)':
E                  No match.
E           
E           During: typing of setitem at /Users/geoffreysametz/anaconda3/anaconda3/envs/nmrsim_39/lib/python3.9/site-packages/sparse/_common.py (969)
E           
E           File "../../../anaconda3/anaconda3/envs/nmrsim_39/lib/python3.9/site-packages/sparse/_common.py", line 969:
E               def _dot_coo_coo(
E                   <source elided>
E                           ):
E                               sums[k] += av * bv
E                               ^

../../../anaconda3/anaconda3/envs/nmrsim_39/lib/python3.9/site-packages/numba/core/dispatcher.py:361: TypingError
```