"""A temporary home for some descriptor classes. These were originally used
to validate class setters. May be deleted in future.
"""
import abc
import numbers
import numpy as np


class AutoStorage:
    """A descriptor class for managing getters and setters.

    See L. Ramalho, "Fluent Python", Ch. 20.
    """
    """Downside of this method is non-informative docstrings for attributes.
    Consider another implementation.
    """
    # TODO: shelve this method and use @property instead for clear attr docs
    __counter = 0

    def __init__(self):
        cls = self.__class__
        prefix = cls.__name__
        index = cls.__counter
        self.storage_name = '_{}#{}'.format(prefix, index)
        cls.__counter += 1

    def __get__(self, instance, owner):
        if instance is None:
            return self
        else:
            return getattr(instance, self.storage_name)

    def __set__(self, instance, value):
        setattr(instance, self.storage_name, value)


class Validated(abc.ABC, AutoStorage):
    """An abstract subclass of AutoStorage that provides validation for
    setters.

    See L. Ramalho, "Fluent Python", Ch. 20.
    """
    def __set__(self, instance, value):
        value = self.validate(instance, value)
        super().__set__(instance, value)

    @abc.abstractmethod
    def validate(self, instance, value):
        """return validated value or raise ValueError"""


class Number(Validated):
    """A descriptor used to validate that a class attribute is a real number.
    """

    def validate(self, instance, value):
        """Verify that value is a real number.

        Parameters
        ----------
        value: The value to be tested.

        Returns
        -------
        value

        Raises
        ------
        TypeError
            If the value is not a real number.
        """
        if not isinstance(value, numbers.Real):
            raise TypeError('value must be a real number')
        return value


class Couplings(Validated):
    """A descriptor used to validate that a value resembles an array of number
    pairs (for each J/# of nuclei entry.
    """
    def validate(self, instance, value):
        """Test that J resembles an array of number pairs (for each J/# of
        nuclei entry.

        Parameters
        ----------
        value: The value to be tested.

        Returns
        -------
        value

        Raises
        ------
        TypeError
            If value is not either an empty array or an array of shape (n, 2).
        """
        testarray = np.array(value)
        if testarray.shape == (0,):  # empty list
            return value
        if len(testarray.shape) != 2:
            print('first entry in array is: ', value[0])
            raise TypeError('J should be 2D array-like')

        _, n = testarray.shape
        if n != 2:
            raise ValueError('J should have a second dimension of 2 '
                             'for J value, # of nuclei.')
        return value
