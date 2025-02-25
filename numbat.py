"""
`numbat`: NumPy+Jax with named axes and an uncompromising attitude

# License

Copyright (C) 2025 Justin Domke - All Rights Reserved

Licensed under the GNU Affero General Public License, Version 3.0 (the "License"); you may not
use this file except in compliance with the License. You may obtain a copy of the License at
https://www.gnu.org/licenses/agpl-3.0.en.html

This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without
even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.

# Requirements

* Python 3.10+
* Numpy
* Jax
* [varname](https://github.com/pwwang/python-varname) (Optional: For magical axis naming.)

# Installation

1. It's a single file.
2. Download it and put it in a directory.
3. Done!

# How do I...

## ...create named tensors?

See `ntensor.__init__`, `ones`, `zeros`, `randn`.

TODO: add rand, others?

Examples
--------
>>> ntensor([0,1,2],'i')
<ntensor {i:3} [0 1 2]>

>>> ones(cats=2, dogs=3)
<ntensor {cats:2, dogs:3}
[[1. 1. 1.]
 [1. 1. 1.]]>

## ...manipulate named tensors?

See `concatenate`, `stack`

## ...do inner/outer/matrix/tensor products or einstein summation?

See `dot`. It does it all. This is the only function you need. Or, equivalently, use the `@` operator which
will call `dot` for you.

Examples
--------
>>> A = ntensor([[0,1],[1, 0]],'i','j')
>>> B = ntensor([5,9],'j')
>>> dot(A,B)
<ntensor {i:2} [9 5]>
>>> dot(B,A)
<ntensor {i:2} [9 5]>
>>> A @ B
<ntensor {i:2} [9 5]>
>>> B @ A
<ntensor {i:2} [9 5]>

## ...do standard scalar operations?

See `abs`, `acos`, `acosh`, ..., `relu`.

Examples
--------
>>> A = ntensor([-3,0,3],'dog')
>>> abs(A)
<ntensor {dog:3} [3 0 3]>
>>> relu(A)
<ntensor {dog:3} [0 0 3]>

## ...do standard pairwise operations?

See `add`, `arctan2`, ..., `true_divide`. Or use operator overloading.

Examples
--------
>>> A = ntensor([0,1,2],'i')
>>> B = ntensor(10)
>>> C = ntensor([10, 20],'j')
>>> add(A, B)
<ntensor {i:3} [10 11 12]>
>>> A+B
<ntensor {i:3} [10 11 12]>
>>> A/C
<ntensor {i:3, j:2}
[[0.   0.  ]
 [0.1  0.05]
 [0.2  0.1 ]]>

## ...do standard reductions?

See `all`, `any`, ..., `var`

Examples
--------
>>> A = ntensor([[True, True, True],[True,False,True]],'i','j')
>>> all(A)
<ntensor {} False>
>>> all(A, axes={'i','j'})
<ntensor {} False>
>>> all(A,axes='i')
<ntensor {j:3} [ True False  True]>
>>> all(A,axes='j')
<ntensor {i:2} [ True False]>

>>> B = ones(i=2, j=3, k=4)
>>> sum(B, axes={'i','k'})
<ntensor {j:3} [8. 8. 8.]>
>>> sum(B, vmap={'j'})
<ntensor {j:3} [8. 8. 8.]>
>>> sum(B, vmap=['i','j'])
<ntensor {i:2, j:3}
[[4. 4. 4.]
 [4. 4. 4.]]>

## ...use standard jax functions?

See `lift`.

Examples
--------
>>> fun = lift(jnp.linalg.solve, 'i j, k->l')
>>> A = ntensor([[2,0],[0,4]],'i','j')
>>> x = ntensor([1, 1], 'k')
>>> fun(A, x)
<ntensor {l:2} [0.5  0.25]>

## ...do linear algebra?

See `inv`, `solve`. Or use `lift`.

## ...index arrays?

See `ntensor.__call__`, which is achieved with *parentheses*, not square brackets.

Examples
--------

>>> dog = axes()
>>> A = ntensor([10, 20, 30, 40, 50],'dog')
>>> A(dog=2)
<ntensor {} 30>
>>> A(dog=dog[1::2])
<ntensor {dog:2} [20 40]>

>>> i, j, k = axes()
>>> A = randn(i=10, j=20, k=30)
>>> A(j=5).shape
ShapeDict(i=10, k=30)
>>> A(j=5, k=3).shape
ShapeDict(i=10)
>>> A(j=j[::2], i=i[::2], k=k[::2]).shape
ShapeDict(i=5, j=10, k=15)

>>> i, j, k = axes()
>>> A = randn(i=10, j=20, k=30)
>>> i_index = ntensor([0, 1, 2], 'l')
>>> j_index = ntensor([0, 1, 2], 'l')
>>> k_index = ntensor([[0, 1, 2, 3],[4, 5, 6, 7]], 'm', 'n')
>>> A(i=i_index, j=j_index, k=k_index).shape
ShapeDict(l=3, m=2, n=4)

## ...take gradients?

See `grad` and `value_and_grad`

Examples
--------
>>> A = ntensor([1., 2., 3.],'i')
>>> B = ntensor([4., 5., 6.],'i')
>>> grad(dot, 0)(A, B)
<ntensor {i:3} [4. 5. 6.]>
>>> grad(dot, 1)(A, B)
<ntensor {i:3} [1. 2. 3.]>


## ...do batched/vmapped operations?

See `batch`, `vmap`, `wrap`, `scan`.

## ...get a plain numpy array?

See `ntensor.numpy`

## ...do signal processing?

See `convolve`


# Key features

1. Numpy provides a huge number of "dot"-type routines, each with their own conventions.
There's [`np.dot`](https://numpy.org/doc/stable/reference/generated/numpy.dot.html#numpy.dot), [`np.inner`](
https://numpy.org/doc/stable/reference/generated/numpy.inner.html#numpy.inner), [`np.outer`](
https://numpy.org/doc/stable/reference/generated/numpy.outer.html#numpy.outer),
[`np.matmul`](https://numpy.org/doc/stable/reference/generated/numpy.matmul.html#numpy.matmul)
[`np.linalg.multi_dot`](https://numpy.org/doc/stable/reference/generated/numpy.linalg.multi_dot.html#numpy
.linalg.multi_dot),
[`np.tensordot`](https://numpy.org/doc/stable/reference/generated/numpy.tensordot.html#numpy.tensordot), and
[`np.einsum`](https://numpy.org/doc/stable/reference/generated/numpy.einsum.html#numpy.einsum).
Each of these has their own dimensional ordering conventions. What does `np.dot(a,b)` do when `a` and
`b` are high-dimensional? Does anyone remember?

   In ntensor, there's only one function, `dot`, and only one (fairly simple) rule to remember.

2. Numpy indexing is comically complex. You've got "basic" indexing, "advanced boolean" indexing,
"advanced vectorized" indexing and "advanced mixed" indexing. And "advanced" indexing has
some confusing special cases that few people understand.

    If you only use slices, integers, and `None` (aka `np.newaxis`), it's *relatively* simple. Writing `x[
    17, None, 3, ..., 2:29:2, None]` is fine. But  what if you want to take the `k`th element from the
    `n`th axis? You're left writing something like

    ```python
    x[*[slice(None) if i != n else k for i in range(x.ndim)]] # eek
    ```

     And advanced indexing is really crazy. I mean, look at this:

    >>> a = onp.ones((50, 60, 70, 80))
    >>> a[[0,1,2],[3,4,5],:,:].shape
    (3, 70, 80)
    >>> a[[0,1,2],:,[3,4,5],:].shape
    (3, 60, 80)
    >>> a[[0,1,2],:,:,[3,4,5]].shape
    (3, 60, 70)
    >>> a[:,[0,1,2],[3,4,5],:].shape # !?
    (50, 3, 80)
    >>> a[:,[0,1,2],:,[3,4,5]].shape
    (3, 50, 70)
    >>> a[:,:,[0,1,2],[3,4,5]].shape # !?
    (50, 60, 3)

    Yes, that is what happens. *This is Numpy behaving **as designed**.*

    Also, Numpy doesn't ([currently](https://numpy.org/neps/nep-0021-advanced-indexing.html)) support
    "outer" indexing, where you'd (say) get each combination of the 2nd/4th/5th rows and 1st/5th columns.
    You can achieve this using utilities
    [`np.ix_`](https://numpy.org/doc/stable/reference/generated/numpy.ix_.html) and
    [`np.ogrid`](https://numpy.org/doc/stable/reference/generated/numpy.ogrid.html) and
    [`np.mgrid`](https://numpy.org/doc/stable/reference/generated/numpy.mgrid.html), but that
    that's even more complexity.

    In ntensor, you don't need any of this stuff. There is just one indexing rule. It's *sort* of like
    always using numpy advanced indexing all the time, except (1) it's really easy because all the axes are
    named, and (2) it's much easier because you don't have to worry about the order and (3) you don't need
    all these crazy utility functions. You can probably understand it just from looking at these examples:

    >>> i, j, k, l, m = axes()
    >>> a = randn(i=10, j=20, k=30, l=40)
    >>> a(k=2).shape
    ShapeDict(i=10, j=20, l=40)
    >>> a(k=2, i=5).shape
    ShapeDict(j=20, l=40)
    >>> idx = ntensor([0,1,2],m)
    >>> a(k=idx).shape
    ShapeDict(i=10, j=20, m=3, l=40)
    >>> a(i=i[0,1,2], j=j[0,1,2]).shape
    ShapeDict(i=3, j=3, k=30, l=40)

3. Manually batching everything is incredibly tedious and error-prone. Need to remember what axis
corresponds to what meaning. Gets out of control with complex cases.


"""

__docformat__ = 'numpy'

import jax
from jax import numpy as jnp
import numpy as onp
from numpy.typing import ArrayLike
from typing import Set, Iterable, Sequence, Callable, Tuple, Union, Dict, Self, Type
from collections import abc
import builtins
from numbers import Number

from pangolin.util import WriteOnceDict

_fundamental_stuff = ['Axis', 'ntensor']
_qol_stuff = ['axes', 'allclose']
_creation_stuff = ['ones', 'zeros', 'randn']
_batching_stuff = ['vmap', 'batch', 'wrap', 'scan']
_lifting_and_lowering_stuff = ['lift', 'lower']
_manipulation_stuff = ['concatenate','stack']

_elementwise_stuff = ['abs', 'acos', 'acosh', 'arccos', 'arccosh', 'arcsin', 'arcsinh', 'arctan', 'arctanh',
                      'asin', 'asinh', 'atan', 'atanh', 'ceil', 'conj', 'cos', 'cosh', 'deg2rad', 'exp',
                      'exp2', 'expm1', 'i0', 'imag', 'log', 'log10', 'log1p', 'log2', 'rad2deg', 'real',
                      'reciprocal', 'rint', 'round', 'sign', 'sin', 'sinc', 'sinh', 'sqrt', 'square', 'tan',
                      'tanh', 'trunc', 'relu']

_pairwise_stuff = ['add', 'arctan2', 'atan2', 'divide', 'divmod', 'equal', 'floor_divide', 'gcd', 'greater',
                   'greater_equal', 'heaviside', 'hypot', 'lcm', 'ldexp', 'less', 'less_equal', 'logaddexp',
                   'logaddexp2', 'maximum', 'minimum', 'multiply', 'mod', 'nextafter', 'not_equal',
                   'polyadd', 'polymul', 'polysub', 'polyval', 'pow', 'power', 'remainder', 'subtract',
                   'true_divide']

_linalg_stuff = ['solve','inv']

_gradient_stuff = ['grad','value_and_grad']

_reductions_stuff = ['all', 'any', 'logsumexp', 'max', 'mean', 'median', 'min',  'prod', 'sum', 'std', 'var']

_other_stuff = ['dot']

__all__ = (
        _fundamental_stuff + _manipulation_stuff + _other_stuff + _qol_stuff + _creation_stuff +
        _batching_stuff +
        _lifting_and_lowering_stuff + _elementwise_stuff + _pairwise_stuff + _reductions_stuff +
        _gradient_stuff + _linalg_stuff)


# Small utility stuff ##############################################################################


class ShapeDict(dict):
    """A ShapeDict is basically just a dict, except:
    1. It is frozen after creation
    2. It ensures that all keys are of type `Axis`
    3. It ensures that all values are of type `int`
    4. It prints more compactly

    Examples
    --------

    >>> d = ShapeDict(i=3)
    >>> d
    ShapeDict(i=3)
    >>> print(d)
    {i:3}
    >>> d['i']
    3
    >>> d[Axis('i')]
    3

    >>> d = ShapeDict({'dogs':3, Axis('cats'):4}, zebras=7, frogs=9)
    >>> d
    ShapeDict(dogs=3, cats=4, zebras=7, frogs=9)
    >>> print(d)
    {dogs:3, cats:4, zebras:7, frogs:9}
    """

    def __setitem__(self, key, value):
        raise NotImplementedError("Can't assign to ShapeDict after creation.")

    def __init__(self, base_dict=None, **kwargs):
        new_dict = {}
        if base_dict is None:
            base_dict = {}
        for ax in base_dict:
            if not isinstance(ax, (str, Axis)):
                raise ValueError(f"ShapeDict keys must be str or Axis, got {ax}")
            if not isinstance(base_dict[ax], int):
                raise ValueError(f"ShapeDict values must be int, got {base_dict[ax]}")
            new_dict[Axis(ax)] = base_dict[ax]

        for ax in kwargs:
            if not isinstance(kwargs[ax], int):
                raise ValueError(f"ShapeDict values must be int, got {kwargs[ax]}")
            if Axis(ax) in new_dict:
                raise ValueError(f"ShapeDict got repeated axis {ax}")
            new_dict[Axis(ax)] = kwargs[ax]

        super().__init__(new_dict)

    def __repr__(self):
        out = 'ShapeDict('
        for key in self:
            out += str(key)
            out += '='
            out += str(self[key])
            out += ', '
        if len(self):
            out = out[:-2]
        out += ')'
        return out

    def __str__(self):
        out = '{'
        for key in self:
            out += str(key)
            out += ':'
            out += str(self[key])
            out += ', '
        if len(self):
            out = out[:-2]
        out += '}'
        return out

    @property
    def axes(self):
        return tuple(self.keys())


class ConsistentDict(dict):
    "like a dict, but upon repeated assignments, enforces equality"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __setitem__(self, key, value):
        if key in self:
            if self[key] != value:
                raise ValueError(
                    f"Tried to assign {key} to {value} but already has value {self[key]}.")
        super().__setitem__(key, value)


def is_set_of_type(obj: object, typ: Type | Sequence[Type], len_zero_ok=True) -> bool:
    if isinstance(obj, (Set,frozenset)):
        if len_zero_ok:
            return builtins.all(isinstance(item, typ) for item in obj)
        else:
            return len(obj) > 0 and builtins.all(isinstance(item, typ) for item in obj)
    return False


def is_sequence_of_type(obj: object, typ: Type | Sequence[Type], len_zero_ok=True) -> bool:
    if isinstance(obj, Sequence):
        if len_zero_ok:
            return builtins.all(isinstance(item, typ) for item in obj)
        else:
            return len(obj) > 0 and builtins.all(isinstance(item, typ) for item in obj)
    return False


def can_be_converted_to_ndarray(obj):
    try:
        jnp.array(obj)  # Try to convert to ndarray
        return True
    except Exception as e:
        return False


def convert_to_0dim_array_if_possible(obj):
    if isinstance(obj, (float, int)):
        return ntensor(obj)
    if can_be_converted_to_ndarray(obj):
        data = jnp.array(obj)
        if data.ndim == 0:
            return ntensor(data)
    return obj


def convert_empty_dict(arg):
    "If arg is the {} dict, get an empty set. Otherwise do nothing."
    if isinstance(arg, dict) and len(arg) == 0:
        return frozenset()
    return arg


def is_ntensor(x):
    return isinstance(x, ntensor)


def ensure_ntensor(x):
    if isinstance(x, ntensor):
        return x
    elif isinstance(x, Number):
        return ntensor(x)
    else:
        return ntensor(x)
        # raise ValueError(f"Couldn't cast {x} to NTensor")


def ensure_ntensor_tree(x):
    return jax.tree.map(ensure_ntensor, x, is_leaf=is_ntensor)

def other_axes(axes, *args):
    leaves = jax.tree.leaves(args, is_leaf=is_ntensor)
    if len(leaves) == 0:
        return axes
    out = []
    for ar in leaves:
        for ax in ar.axes:
            if ax not in out and ax not in axes:
                out.append(ax)
    return tuple(out)

# The fundamental classes ####################################################################################


class Axis:
    """An axis is a unique axis label. It's *basically* just a string but with a few tricks to make
    notation nicer. For convenience, axes evaluate as to equal to their strings. Axes are equal based on
    names, not object identity.

    Examples
    --------
    >>> cat = Axis('cat')
    >>> dog = Axis('dog')
    >>> cat == dog
    False
    >>> cat == 'cat'
    True
    >>> cat == 'dog'
    False

    """

    _name: str
    "The stored name. Don't change this!"

    def __init__(self, name: None | str | Self = None):
        """Create an `Axis`.

        Parameters
        ----------
        name:
            Create an axis with the given name. If `None`, then try to detect name being assigned to
            with evil magics.

        Examples
        --------
        >>> time = Axis('time')
        >>> time
        Axis('time')
        >>> time._name
        'time'
        >>> time == 'time' # can compare to strings
        True
        >>> time == Axis('time') # can compare to other Axis with same name
        True
        >>> id(time) == id(Axis('time')) # even though different objects
        False

        >>> days = Axis() # evil magic
        >>> days._name
        'days'
        >>> days == 'days'
        True
        >>> days == Axis('days')
        True

        >>> alice = Axis("bob") # this is almost certainly a bad idea
        >>> alice._name
        'bob'
        >>> alice == 'bob'
        True
        >>> alice == 'alice'
        False

        """
        if name is None:
            # try to detect name using evil magics
            import varname
            name = varname.varname()

        if isinstance(name, str):
            self._name = name
        elif isinstance(name, Axis):
            self._name = name._name
        else:
            raise TypeError(f"name must be str or Axis, got {name}")

    def __eq__(self, other) -> bool:
        if isinstance(other, Axis):
            return self._name == other._name
        elif isinstance(other, str):
            return self._name == other
        else:
            raise TypeError(f"name must be str or Axis, got {other}")

    def __hash__(self) -> int:
        return hash(self._name)

    def __str__(self) -> str:
        return self._name

    def __repr__(self) -> str:
        return f"Axis('{self._name}')"

    def __getitem__(self, item: slice | ArrayLike):
        """@public

        You can index into an `Axis` for nicer syntactic sugar when indexing an `ntensor`. If `x` is
        an `ntensor` with some `Axis` `ax`, then if you do `x(ax=ax[(1,2,3)])`, that's equivalent to
        `x(ax=ntensor((1,2,3),'ax'))`.

        If you do `x(ax=ax[1:10:3])`, that's equivalent to `x(ax=IndexedAxis(ax,slice(1,10,3)))` which is
        in turn equivalent to `x(ax=NTensor(np.arange(x.shape['ax'])[1:10:3])`.

        This is achieved by making `ax[...]` do slightly tricky stuff, which is this function.

        Parameters
        ----------
        item: slice | ArrayLike
            If this is a slice, then returns an `IndexedAxis`. If this is anything that can be cast
            to a 1D numpy array, then this returns an `ntensor`. Anything else is an error.

        Examples
        --------
        >>> cats = Axis('cats')

        OK: Numpy arrays or things castable to them.

        >>> cats[jnp.array([7,5,2,9])]
        <ntensor {cats:4} [7 5 2 9]>
        >>> cats[(7,5,2,9)]
        <ntensor {cats:4} [7 5 2 9]>
        >>> cats[7,5,2,9]
        <ntensor {cats:4} [7 5 2 9]>

        Also OK: slices.

        >>> cats[:]
        IndexedAxis(cats, slice(None, None, None))
        >>> cats[1:20:3]
        IndexedAxis(cats, slice(1, 20, 3))

        Not OK: Integers. (Since 0-dimensional arrays don't have shapes, this has no purpose.)

        >>> cats[0]
        Traceback (most recent call last):
        ...
        ValueError: Axis index castable to numpy array, but got 0 dims (must be 1)

        Not OK: `ntensor` (What would `x(ax=ax[y])` mean if `y` is `ntensor`? Allowing this would lead to
        confusion.)

        >>> cats[ntensor(3)]
        Traceback (most recent call last):
        ...
        ValueError: Can't index an Axis with an NTensor (1-D numpy array OK).

        Not OK: Thing castable to 2+ dimensional numpy arrays. (We only have 1 `Axis`!)

        >>> cats[((1,2,3),(4,5,6))] # 2d index not allowed!
        Traceback (most recent call last):
        ...
        ValueError: Axis index castable to numpy array, but got 2 dims (must be 1)


        """

        if isinstance(item, ntensor):
            raise ValueError("Can't index an Axis with an NTensor (1-D numpy array OK).")

        if isinstance(item, slice):
            return IndexedAxis(self, item)
        else:
            try:
                data = jnp.array(item)
            except ValueError:
                raise ValueError(
                    f"Axis index ({item}) not slice and not castable to a 1-d numpy array")
            if data.ndim != 1:
                raise ValueError(f"Axis index castable to numpy array, but got {data.ndim} dims (must be 1)")
            return ntensor(item, self)

    def __lt__(self, other) -> bool:
        """Make axes sortable by name (mostly just to make tests deterministic)"""
        return self._name < other._name


class IndexedAxis:
    def __init__(self, axis: Axis, index: slice | ArrayLike):
        """
        An axis and an index into it. This class just stores the

        Parameters
        ----------
        axis: Axis
            The axis being indexed. Needs to be an actual Axis, not a str.
        index
            The indices into the axis. Could be slice or something that can be cast to a 1-D
            numpy array. Cannot be NTensor!
        """
        assert isinstance(axis, Axis)
        self.axis = Axis(axis)
        self.index = index

    def __hash__(self) -> int:
        return hash((self.axis, self.index))

    def __eq__(self, other) -> bool:
        return isinstance(other,
                          IndexedAxis) and self.axis == other.axis and self.index == other.index

    def __repr__(self):
        return f"IndexedAxis({self.axis}, {self.index})"


class ntensor:
    """An array is the base type of this package. The first thing about `array`s is that you can
    create them. The second thing about `array`s is that you can (elegantly) index them. The third
    thing about `array`s is you can call functions on them.

    For core usage, see:
    * `__init__` for creating arrays
    * `axes` for getting the axes
    * `__call__` for indexing
    * `__getitem__` for converting to

    """

    _data: jnp.ndarray
    "The underlying data, stored as a numpy array. (You shouldn't need to mess with this directly!)"

    _axes_ordered: Tuple[Axis, ...]
    "The Axis for each dimension in `data`. (You shouldn't need to mess with this directly!)"

    def __init__(self, data: ArrayLike, *axes: str | Axis):
        """Create an array from some block of data. This is familiar from other array languages,
        except you must provide axis names if there are more than 0 dimensions

        Parameters
        ----------
        data
            An array or  [ArrayLike](https://docs.jax.dev/en/latest/_autosummary/jax.typing.ArrayLike.html)
            that can be cast to a numpy array.
        *axes
            `Axis` or strings. Must have same length as number of dimensions of data.

        Examples
        --------
        Zero dimensional arrays have no dimensions, and so don't have axis names

        >>> ntensor(137.036)
        <ntensor {} 137.036>

        Arrays with at least one dimension must have axis names

        >>> ntensor([11, 22, 33], 'cats')
        <ntensor {cats:3} [11 22 33]>
        >>> ntensor([5.5], 'i')
        <ntensor {i:1} [5.5]>
        >>> ntensor([[1,2,3],[4,5,6]], 'row', 'col')
        <ntensor {row:2, col:3}
        [[1 2 3]
         [4 5 6]]>

        You can create an explicit `Axis` objects first if you want. This is *completely equivalent*. (If you
        pass strings, they are automatically converted into `Axis` object for you.
        >>> cats = Axis('cats')
        >>> ntensor([11, 22, 33], cats)
        <ntensor {cats:3} [11 22 33]>
        """
        axes = tuple(Axis(ax) for ax in axes)

        if len(set(axes)) != len(axes):
            raise ValueError(f"ntensor got non-unique axes {axes}")

        #print(f"about to create data with array {data=}")
        self._data = jnp.array(data)

        if self._data.ndim != len(axes):
            raise ValueError(
                f"data has shape {self._data.shape} with {self._data.ndim} dims, but given "
                f"{len(axes)} axes: {tuple(str(ax) for ax in axes)}")

        self._axes_ordered = axes

        #self._shape = ShapeDict({axes[i]: self._data.shape[i] for i in range(len(axes))})

        # # hack to make jax.grad work for scalars
        # if self._data.ndim == 0:
        #     self.__jax_array__ = lambda: self._data
        #     self.dtype = self._data.dtype

    def __jax_array__(self):
        if self.ndim > 1:
            raise ValueError("Only ntensor with <= 1 can be auto-converted to jax array")
        return self._data

    def __array__(self, dtype=None):
        if self.ndim > 1:
            raise ValueError("Only ntensor with <= 1 can be auto-converted to array")
        print(f"__array__ {self=} {dtype=}")
        return onp.asarray(self._data, dtype=dtype)

    @property
    def dtype(self):
        return self._data.dtype

    @property
    def axes(self) -> frozenset[Axis]:
        """The Axis objects. It's a set, because axes *don't have an order* (that you're supposed to
        think about).

        Examples
        --------
        >>> A = ntensor([[1,2,3],[4,5,6]], "day", "person")
        >>> A.axes == {Axis('day'), Axis('person')}
        True
        >>> A.axes == {'day', 'person'}
        True
        """
        return frozenset(self._axes_ordered)

    # @property
    # def _axes_ordered(self) -> Tuple[Axis, ...]:
    #     return tuple(self._shape.keys())

    @property
    def ndim(self) -> int:
        """The number of dimensions. Equal to `len(axes)`

        Examples
        --------
        >>> A = ntensor([[1,2,3],[4,5,6]],'rows','cols')
        >>> A.ndim
        2
        """
        return len(self.axes)

    @property
    def shape(self) -> ShapeDict:
        """The shape is a *dict* mapping axes to ints.

        Examples
        --------
        >>> A = ntensor([[1,2,3],[4,5,6]],'record','hour')
        >>> A.shape
        ShapeDict(record=2, hour=3)
        >>> print(A.shape)
        {record:2, hour:3}
        """
        return ShapeDict({self._axes_ordered[i]: self._data.shape[i] for i in range(self.ndim)})

    @property
    def _shape_ordered(self):
        return tuple(self.shape[i] for i in self._axes_ordered)

    def __repr__(self):
        if self.ndim == 0:
            return "<ntensor {} " + str(self._data) + ">"
        elif self.ndim == 1:
            return f"<ntensor {self.shape} {str(self._data)}>"
        else:
            out = f"<ntensor {self.shape}\n"
            out += str(self._data) + '>'
        return out

    def numpy_broadcasted(self, *axes: Axis | str, strict=False) -> jnp.ndarray:
        """Convert to a (Jax) numpy array, by providing order for axes.
        If you provide an axis that doesn't exist, a new one with a singleton dimension
        will be created

        Parameters
        ----------
        *axes
            The axes, in order. Must include all axes present in self.

        Examples
        --------

        Zero dimensional array

        >>> A = ntensor(5)
        >>> A
        <ntensor {} 5>
        >>> print(A.numpy_broadcasted())
        5
        >>> print(A.numpy_broadcasted('i'))
        [5]
        >>> print(A.numpy_broadcasted('i','j'))
        [[5]]

        Two-dimensional array

        >>> A = ntensor([[1,2,3],[4,5,6]],'i','j')
        >>> print(A.numpy_broadcasted('i','j'))
        [[1 2 3]
         [4 5 6]]
        >>> print(A.numpy_broadcasted('j','i'))
        [[1 4]
         [2 5]
         [3 6]]
        >>> print(A.numpy_broadcasted('i','k','j','l').shape)
        (2, 1, 3, 1)
        """

        axes = tuple(Axis(ax) for ax in axes)

        if not (self.axes <= set(axes)):
            raise ValueError(f"axes {self.axes} not a subset of {set(axes)}")

        if strict and len(axes) != self.ndim:
            raise ValueError(f"{len(axes)=} not equal to {self.ndim=}")


        extra_dims = len(set(axes) - self.axes)
        data = self._data.reshape(self._data.shape + (1,) * extra_dims)
        where = []
        where_extra = self._data.ndim
        for ax in axes:
            if ax in self.axes:
                where.append(self._axes_ordered.index(ax))
            else:
                where.append(where_extra)
                where_extra = where_extra + 1

        return data.transpose(where)

    def numpy(self, *axes: Axis | str, strict=False) -> jnp.ndarray:
        """Convert to a (Jax) numpy array, by providing order for axes.

        Unlike numpy_broadcasted, you cannot provide new axes. Also, if self has only 1 dimension,
        it's acceptable to skip it.

        Parameters
        ----------
        *axes
            The axes, in order. Must have equal number as dimensions, unless number of dimensions is one,
            in which case it can be skipped

        Examples
        --------

        Zero dimensional array

        >>> A = ntensor(5)
        >>> A
        <ntensor {} 5>
        >>> print(A.numpy())
        5

        One dimensional array
        >>> A = ntensor([1,2,3],'i')
        >>> print(A.numpy('i'))
        [1 2 3]
        >>> print(A.numpy())
        [1 2 3]

        Two-dimensional array

        >>> A = ntensor([[1,2,3],[4,5,6]],'i','j')
        >>> print(A.numpy('i','j'))
        [[1 2 3]
         [4 5 6]]
        >>> print(A.numpy('j','i'))
        [[1 4]
         [2 5]
         [3 6]]
        """

        if axes == () and self.ndim == 1:
            axes = (self._axes_ordered[0],)

        axes = tuple(Axis(ax) for ax in axes)

        if set(axes) != self.axes:
            raise ValueError(f"Provided axes {axes} do not match ntensor axes {self.axes}")


        extra_dims = len(set(axes) - self.axes)
        data = self._data.reshape(self._data.shape + (1,) * extra_dims)
        where = []
        where_extra = self._data.ndim
        for ax in axes:
            if ax in self.axes:
                where.append(self._axes_ordered.index(ax))
            else:
                where.append(where_extra)
                where_extra = where_extra + 1

        return data.transpose(where)



    # def __getitem__(self, idx):
    #     if isinstance(idx, slice):
    #         raise ValueError("Can't index NTensor with slice. (Remember, A[i,j] converts to numpy)")
    #
    #     if isinstance(idx, int):
    #         raise ValueError("Can't index NTensor with int. (Remember, A[i] converts to numpy)")
    #
    #     if idx is ...:
    #         if self.shape != {}:
    #             raise ValueError(
    #                 f"Can only call A[...] for scalar NTensor, got shape {self.shape})")
    #         else:
    #             return self.numpy_broadcasted()
    #
    #     if isinstance(idx, (str, Axis)):
    #         return self.numpy_broadcasted(idx)
    #
    #     if len(idx) != self.ndim:
    #         raise ValueError(
    #             f"Number of indices in {idx} not equal to number of dimensions ({self.ndim})")
    #
    #     return self.numpy_broadcasted(*idx)

    def __add__(self, other: Self | Number) -> Self:
        """@public
        Do fully-batched addition. Unlike `add`, this seamlessly handles arrays of any shape with
        full automatic broadcasting.

        `A+B` is equivalent to `add(self, other)`, i.e. maps over all axes.

        Examples
        -------
        >>> A = ntensor([0,1,2],'cats')
        >>> B = ntensor([0,5],'dogs')
        >>> (A+B).shape == {'cats': 3, 'dogs':2}
        True
        """

        return add(self, other)

    def __radd__(self, other) -> 'ntensor':
        return add(other, self)

    def __sub__(self, other) -> 'ntensor':
        """@public
        Do fully-batched substraction.
        """
        return subtract(self, other)

    def __rsub__(self, other) -> 'ntensor':
        return subtract(other, self)

    def __mul__(self, other) -> 'ntensor':
        """@public
        Do fully-batched multiplication.
        """
        return multiply(self, other)

    def __rmul__(self, other) -> 'ntensor':
        return multiply(other, self)

    def __truediv__(self, other) -> 'ntensor':
        """@public
        Do fully-batched division.
        """
        return true_divide(self, other)

    def __rtruediv__(self, other) -> 'ntensor':
        return true_divide(other, self)

    def __pow__(self, other) -> 'ntensor':
        """@public
        Do fully-batched powers.
        """
        return power(self, other)

    def __rpow__(self, other) -> 'ntensor':
        return power(other, self)

    # def __float__(self)->float:
    #     if self.shape != {}:
    #         raise ValueError(f"Can't convert NTensor with shape {self.shape} to float")
    #     return float(self._data)

    # def __array__(self)->jnp.ndarray:
    #     print("ARRAY CALLED")
    #     if self.shape != {}:
    #         raise ValueError(f"Can't convert NTensor with shape {self.shape} to array")
    #     return self._data
    #
    # @property
    # def dtype(self):
    #     print("DTYPE CALLED")
    #     return self._data.dtype
    #
    # @property
    # def aval(self):
    #     print("AVAL CALLED")
    #     #return self._data.aval
    #     return jax.core.ShapedArray((),dtype=self._data.dtype)
    #     #return jnp.ones(())

    # @property
    # def T(self):
    #     return NTensor(self._data, *reversed(self._axes_ordered))
    #
    def __matmul__(self, other: 'ntensor') -> 'ntensor':
        """@public

        Do einstein summation on two arrays, taking reductions over all shared axes. This is equivalent to
        an alias for `dot(self, other)`.

        Parameters
        ----------
        other: ntensor

        Returns
        -------
        result: ntensor
            result of Einstein summation on two arrays.

        Examples
        --------
        >>> A = ntensor([0,1,2],'i')
        >>> B = ntensor([5,1,3],'i')
        >>> A @ B
        <ntensor {} 7>

        >>> A = ntensor([0,1],'i')
        >>> B = ntensor([0,2],'j')
        >>> A @ B
        <ntensor {i:2, j:2}
        [[0 0]
         [0 2]]>


        See Also
        --------
        dot
        """
        return dot(self, other)

    def __rmatmul__(self, other: 'ntensor') -> 'ntensor':
        return dot(other, self)

    # def __getitem__(self,idx)->'array':
    #     if not isinstance(idx,tuple):
    #         idx = (idx,)
    #     assert builtins.all(isinstance(i,IndexedAxis) for i in idx), "all inputs must be
    #     IndexedAxis"
    #     return select(self, **{i.axis.name:i.index for i in idx})

    def __call__(self, **kwargs) -> 'ntensor':
        """@public

        Index this array. All indexing follows the One True Rule:
        * All axes must be indexed.
        * All indices are themselves `arrays`.
        * The axes of the indices determine the output axes.
        * Non-shared axes in indices are orthogonal.
        * Shared axes in indices are pointwise. (And must therefore have equal lengths!)

        That's it. That's the only *real* rule. (Cf. numpy's [insanely complex advanced indexing
        rules](https://numpy.org/doc/stable/user/basics.indexing.html#advanced-indexing).) But to
        make things nicer, we offer some ergonomic syntactic sugar.

        1. If you don't explicitly index an input axis, it is implicitly fully sliced,
        with itself as an output axis.
        2. You can slice an input axis and then use that as an index.
        3. You can index any axis (input or not) with a slice or 1-d list/tuple/numpy array. This
        is treated like a 1-d array with that axis.

        Parameters
        ----------
        kwargs

        Returns
        -------
        out: ntensor

        Examples
        --------

        You *can't* index arrays without giving names for axes. If you try, you get an error. (I
        *told* you this package was uncompromising, right?)

        >>> A = ntensor([15,30,45],'i')
        >>> A[1] # Nope. No! Doesn't work!
        Traceback (most recent call last):
        ...
        ValueError: Can't index NTensor with int. (Remember, A[i] converts to numpy)

        You do it like this, always with keywork arguments for axes.

        >>> A = ntensor([15,30,45],'i')
        >>> A(i=1) # yes!
        <ntensor {} 30>

        Here are examples illustrating the one true rule

        >>> A = randn(i=5, j=6)
        >>> i_index = ntensor([1,0],'n')
        >>> j_index = ntensor([2,3],'n')
        >>> B = A(i=i_index, j=j_index)
        >>> print(B.shape)
        {n:2}
        >>> B(n=0) == A(i=1, j=2)
        <ntensor {} True>
        >>> B(n=1) == A(i=0, j=3)
        <ntensor {} True>

        >>> A = randn(i=5, j=6)
        >>> i_index = ntensor([1,0],'n')
        >>> j_index = ntensor([2,3],'m')
        >>> B = A(i=i_index, j=j_index)
        >>> print(B.shape)
        {n:2, m:2}
        >>> B(n=0, m=0) == A(i=1, j=2)
        <ntensor {} True>
        >>> B(n=0, m=1) == A(i=1, j=3)
        <ntensor {} True>
        >>> B(n=1, m=0) == A(i=0, j=2)
        <ntensor {} True>
        >>> B(n=1, m=1) == A(i=0, j=3)
        <ntensor {} True>

        >>> A = randn(i=20, j=20, k=20)
        >>> i_index = ntensor([[0,1,2],[3,4,5]],'k','l')
        >>> j_index = ntensor([6,7,8],'l')
        >>> k_index = ntensor([9,10,11,12],'m')
        >>> B = A(i=i_index, j=j_index, k=k_index)
        >>> print(B.shape)
        {k:2, l:3, m:4}

        You can re-use input as output dimensions if you want. This changes nothing! It's
        equivalent to using all new dimensions and then relabeling.

        >>> i_index = ntensor([[0,1,2],[3,4,5]],'k','i')
        >>> j_index = ntensor([6,7,8],'i')
        >>> k_index = ntensor([9,10,11,12],'m')
        >>> B = A(i=i_index, j=j_index, k=k_index)
        >>> print(B.shape)
        {k:2, i:3, m:4}

        Examples using syntacic sugar for numbers or 1-d lists/tuple/numpy arrays or slices as
        indices

        >>> A = ntensor([15,30,45,60,75,90,105,120],'i')
        >>> A(i=2)                    # ints work
        <ntensor {} 45>
        >>> A(i=[2,0,5])              # lists work
        <ntensor {i:3} [45 15 90]>
        >>> A(i=jnp.array([2,0,5]))    # numpy arrays work
        <ntensor {i:3} [45 15 90]>
        >>> i = Axis('i')
        >>> A(i=i[6:1:-2])
        <ntensor {i:3} [105  75  45]>

        Remember, using numbers or 1D lists/tuples/numpy arrays or slices is just syntactic sugar!

        >>> A = ntensor([15,30,45,60,75,90,105,120],'i')
        >>> A(i=2)==A(i=ntensor(2))
        <ntensor {} True>
        >>> A(i=jnp.array([2,0,5])) == A(i=ntensor(jnp.array([2,0,5]),'i'))
        <ntensor {i:3} [ True  True  True]>
        >>> i = Axis('i')
        >>> A(i=i[6:1:-2]) == A(i=ntensor([6,4,2],'i'))
        <ntensor {i:3} [ True  True  True]>

        You can index an Axis as a shortcut to create a new array.

        >>> A = ntensor([15,30,45,60,75,90,105,120],'i')
        >>> i, j = axes()
        >>> print(A(i=i[:]).shape)
        {i:8}
        >>> A(i=i[:]) == A
        <ntensor {i:8} [ True  True  True  True  True  True  True  True]>
        >>> print(A(i=j[[2,0,5]]).shape)
        {j:3}
        >>> A(i=j[[2,0,5]]) == A(i=ntensor([2,0,5],j))
        <ntensor {j:3} [ True  True  True]>

        You can also index with a *slice* from another input index. (You can *only* slice axes that exist
        in the `NTensor` since otherwise the shape would be ambiguous.)



        """
        return select(self, **kwargs)

    def __lshift__(self, other: Iterable[str | Axis]) -> 'ntensor':
        other = {Axis(o) for o in other}
        assert self.axes == other, f"{self.axes} != {other}"
        return self

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ntensor):
            raise ValueError(f"Can't compare NTensor with {type(other)}")
        if self.shape != other.shape:
            raise ValueError(
                f"Can't compare NTensor with shape {self.shape} to other with shape {other.shape}")
        return equal(self, other)



AxisIndices = ntensor | slice | int | Sequence[int] | ArrayLike | IndexedAxis
AxisLike = str | Axis
AxisSetLike = AxisLike | Sequence[AxisLike] | Set[AxisLike]


# Register NTensor as a pytree node ################################################################

def ntensor_flatten(x: ntensor):
    # print(f"flattening {x=}")
    children = (x._data,)
    aux_data = (x._axes_ordered,)
    return children, aux_data


def ntensor_unflatten(aux_data, children):
    # Do unflattening without calling ntensor.__init__ to avoid issues discussed here:
    # https://docs.jax.dev/en/latest/working-with-pytrees.html
    # in "Custom pytrees and initialization with unexpected values" section
    #print(f"unflattening {aux_data=} {children=}")
    data, = children
    axes_ordered, = aux_data
    obj = object.__new__(ntensor)
    obj._data = data
    obj._axes_ordered = axes_ordered
    return obj

# Global registration
jax.tree_util.register_pytree_node(ntensor, ntensor_flatten, ntensor_unflatten)


# def ntensor_flatten(x: ntensor):
#     # print(f"flattening {x=}")
#     children = ()
#     aux_data = x._data, x._axes_ordered
#     return children, aux_data
#
#
# def ntensor_unflatten(aux_data, children):
#     # print(f"unflattening {aux_data=} {children=}")
#     #x, = children
#     data, axes = aux_data
#     return ntensor(data, *axes)
#
# # Global registration
# jax.tree_util.register_pytree_node(ntensor, ntensor_flatten, ntensor_unflatten)
#



# Various quality of life improvements #############################################################


def axes(*names: str):
    """Convenience function to create a bunch of axes at once

    Parameters
    ----------
    *names: str
        names to give to each element. pass none if you want them to be inferred

    Returns
    -------
    out_axes: tuple[Axis]
        sequence of tuples

    Examples
    --------
    >>> rows, cols, batch = axes('rows', 'cols', 'batch') # safe boring redudnancy
    >>> rows._name
    'rows'
    >>> cols._name
    'cols'
    >>> batch._name
    'batch'

    >>> rows, cols, batch = axes() # exciting evil magic
    >>> rows._name
    'rows'
    >>> cols._name
    'cols'
    >>> batch._name
    'batch'
    """
    if names == ():
        import varname
        names = varname.varname(multi_vars=True)
        if len(names)==1:
            return Axis(names[0])
        else:
            return tuple(Axis(name) for name in names)
    return tuple(Axis(name) for name in names)


def allclose(*arrays: ntensor, rtol=1e-5, atol=1e-8, equal_nan=False) -> bool:
    # TODO: numpy allclose can only handle 2 inputs
    if len(arrays) != 2:
        raise ValueError("2 inputs only")
    axes = arrays[0].axes
    for a in arrays:
        assert a.axes == axes, "all axes must be same"
    axes = arrays[0]._axes_ordered
    numpy_arrays = [a.numpy_broadcasted(*axes) for a in arrays]
    rez = jnp.allclose(*numpy_arrays, rtol=rtol, atol=atol, equal_nan=equal_nan)
    if isinstance(rez, bool):
        return rez
    else:
        assert rez.ndim == 0
        assert rez.dtype == bool
        return bool(rez)


# Creating arrays # ################################################################################

def ones(dtype=None, **axes: int) -> ntensor:
    """Convenience function, over just using `np.ones` and then `ntensor`.

    Parameters
    ----------
    **axes: Dict[str,int]
        Mapping from axes to sizes. Will be converted to axes.

    Returns
    -------
    A: array
        Array with given shape

    Examples
    --------
    >>> A = ones(i=5)
    >>> print(A.shape)
    {i:5}
    >>> B = ones(cats=5, dogs=15)
    >>> print(B.shape)
    {cats:5, dogs:15}
    """
    data = jnp.ones([axes[i] for i in axes],dtype=dtype)
    return ntensor(data, *axes)

def ones_like(A: ntensor, dtype=None) -> ntensor:
    if dtype is None:
        dtype = A.dtype

    str_shape = {str(ax):A.shape[ax] for ax in A._axes_ordered}
    return ones(dtype=dtype, **str_shape)

def zeros(**axes: int) -> ntensor:
    """Convenience function, over just using `np.zeros` and then `array`. Use just like `ones`
    """

    data = jnp.zeros([axes[i] for i in axes])
    return ntensor(data, *axes)

def randn(**kwargs: int):
    """Convenience function, to call `np.random.randn` and then `ntensor`. Use just like
    `ones`

    Parameters
    ----------
    **kwargs: int
        keyword arguments mapping strings to ints

    Returns
    -------
    out: `ntensor`
        random data with requested shape

    Examples
    --------
    >>> A = randn(i=5, j=10, k=100)
    >>> A.shape
    ShapeDict(i=5, j=10, k=100)

    """
    sizes = [kwargs[i] for i in kwargs]
    axes = [Axis(i) for i in kwargs]
    return ntensor(onp.random.randn(*sizes), *axes)

# Random key management ######################################################################################

def PRNGKey(seed:int, axis:str|Axis):
    return ntensor(jax.random.PRNGKey(seed), axis)

def random_split(key, n=None, new_axis=None):
    assert key.ndim == 1
    old_axis, = key.axes
    jax_key = key.numpy_broadcasted(old_axis)

    if n is None and new_axis is None:
        jax_a, jax_b = jax.random.split(jax_key)
        return ntensor(jax_a, old_axis), ntensor(jax_b, old_axis)
    elif n and new_axis:
        jax_keys = jax.random.split(jax_key, n)
        return ntensor(jax_keys, new_axis, old_axis)
    else:
        raise ValueError("either both n and new_axis must be none or both specified")

def random_int(key, minval, maxval):
    if key.ndim != 1:
        raise ValueError("only 1D arrays are supported")
    return ntensor(jax.random.randint(key.numpy(), shape=(), minval=minval, maxval=maxval))

def random_normal(key):
    if key.ndim != 1:
        raise ValueError("only 1D arrays are supported")
    return ntensor(jax.random.normal(key.numpy()))

# Mangling arrays # ##########################################################################################

def concatenate(arrays: Sequence[ntensor], axis:str|Axis) -> ntensor:
    """Concatenate a sequence of NTensors into a NTensor.

    Parameters
    ----------
    arrays: Iterable[ntensor]
        The tensors must have the same shape except in the dimension axis
    axis: str|Axis
        The axis along which to be joined

    Returns
    -------
    out: ntensor
        The concatenated ntensor

    See Also
    --------
    stack

    Examples
    --------
    >>> A = ntensor([1,2],'i')
    >>> B = ntensor([3,4],'i')
    >>> concatenate([A,B], axis='i')
    <ntensor {i:4} [1 2 3 4]>

    >>> A = ntensor([[1,2],[3,4]],'i','j')
    >>> B = ntensor([[5,6],[7,8]],'i','j')
    >>> concatenate([A,B],axis='i')
    <ntensor {j:2, i:4}
    [[1 3 5 7]
     [2 4 6 8]]>
    >>> concatenate((A,B),axis=Axis('i'))
    <ntensor {j:2, i:4}
    [[1 3 5 7]
     [2 4 6 8]]>
    >>> concatenate([A,B],axis='j')
    <ntensor {i:2, j:4}
    [[1 2 5 6]
     [3 4 7 8]]>
    """
    axis = Axis(axis)
    return lift(jnp.concatenate, in_axes=[axis], out_axes=[axis])(arrays)

def stack(arrays: Iterable[ntensor], axis:str|Axis) -> ntensor:
    """Join a squence of ntensor along a new axis into a new ntensor.

    Parameters
    ----------
    arrays: Iterable[ntensor]
        The tensors must have the same shape
    axis: str|Axis
        The axis along which to be joined

    Returns
    -------
    out: ntensor
        The concatenated ntensor

    See Also
    --------
    concatenate

    Examples
    --------
    >>> A = ntensor(5)
    >>> B = ntensor(9)
    >>> stack([A,B],'i')
    <ntensor {i:2} [5 9]>

    >>> A = ntensor([1,2,3],'i')
    >>> B = ntensor([4,5,6],'i')
    >>> stack([A,B], axis='j')
    <ntensor {i:3, j:2}
    [[1 4]
     [2 5]
     [3 6]]>

    >>> A = ntensor([1,2,3],'i')
    >>> B = ntensor([4,5,6],'i')
    >>> stack((A,B), axis='j')
    <ntensor {i:3, j:2}
    [[1 4]
     [2 5]
     [3 6]]>
    """
    axis = Axis(axis)
    return lift(jnp.stack, in_axes=[], out_axes=[axis])(arrays)


# Lifting  #########################################################################################


def parse_lift_str(axes_str, in_axes, out_axes):
    """Parse a lift string

    Parameters
    ----------
    axes_str: str

    Returns
    -------
    in_axes: Tuple[Axis] | Tuple[Tuple[Axis]]
        the in axes

    """

    if axes_str is not None and (in_axes is not None or out_axes is not None):
        raise ValueError(f"Can't provide both {axes_str=} and {in_axes=}/{out_axes=}")

    if axes_str is None and in_axes is None and out_axes is None:
        return [], []
    elif axes_str is None:
        return in_axes, out_axes

    if '->' not in axes_str:
        axes_str = axes_str + '->'

    lhs, rhs = axes_str.split('->')

    def get_ax_tuple(part):
        return tuple(Axis(ax) for ax in part.split())

    def my_split(stuff):
        if ',' in stuff:
            return tuple(get_ax_tuple(part) for part in stuff.split(','))
        else:
            return get_ax_tuple(stuff)

    return my_split(lhs), my_split(rhs)


def lift_pytree(fun: Callable, in_axes=(), out_axes=()) -> Callable:
    """Lower a namedtensor function into a function that operates on jax arrays.

    Parameters
    ----------
    fun
        the function to be lifted
    in_axes
        Input axes. This can be A sequence of sequences of Axis, of the same length as the number of
        args. Or, it can just be a sequence of Axis, in which case all NTensor arguments are mapped
        over the same args.
    out_axes
        The output axes

    Returns
    -------
    new fun
        keyword args are passed unchanged.
    """

    def wrapped(*ntensor_args):
        #ntensor_args = ensure_ntensor_tree(ntensor_args)

        def my_array(in_axes, arg):
            if in_axes is None:
                return arg
            ntensor_arg = ensure_ntensor(arg)

            return ntensor_arg.numpy_broadcasted(*in_axes)

        args = tree_map_axes(my_array, in_axes, ntensor_args)
        out = fun(*args)

        def my_ntensor(out_axes, numpy_out):
            return ntensor(numpy_out, *out_axes)

        return tree_map_axes(my_ntensor, out_axes, out)

    return wrapped


def lift(jax_fun: Callable, axes_str: str | None = None, in_axes=None, out_axes=None,
         batched=True) -> Callable:
    in_axes, out_axes = parse_lift_str(axes_str, in_axes, out_axes)
    fun = lift_pytree(jax_fun, in_axes, out_axes)
    if batched:
        if is_sequence_of_type(in_axes, Axis):
            axes = set(in_axes)
        else:
            axes = set.union(*[set(in_ax) for in_ax in in_axes])
        return batch(fun, axes)
    else:
        return fun

# def abstract_lift(jax_fun, in_ints:Tuple[Tuple[int]], out_ints:Tuple[Tuple[int]])->Callable:
#     def fun(*args):
#         int2ax = WriteOnceDict()
#         for arg, my_in_ints in zip(args, in_ints, strict=True):
#             for ax, i in zip(arg._axe)


# Lowering #########################################################################################


# def tree_map_axes(f, axes, *other):
#     def mini_eval(leaf, *remaining_subtrees):
#         return jax.tree_map(lambda *leaves: f(leaf, *leaves), *remaining_subtrees,
#                             is_leaf=is_ntensor)
#
#     is_axis_sequence = lambda x: is_sequence_of_type(x, Axis)
#     return jax.tree_util.tree_map(mini_eval, axes, *other, is_leaf=is_axis_sequence)


def tree_map_axes(f, axes, *other):
    def mini_eval(leaf, *remaining_subtrees):
        return jax.tree_map(lambda *leaves: f(leaf, *leaves), *remaining_subtrees,
                            is_leaf=lambda x: is_ntensor(x) or x is None)

    is_axis_sequence = lambda x: x is None or is_sequence_of_type(x, Axis)
    return jax.tree_util.tree_map(mini_eval, axes, *other, is_leaf=is_axis_sequence)


def lower_pytree(fun: Callable, in_axes_pytree, out_axes_pytree) -> Callable:
    """Lower a namedtensor function into a function that operates on jax arrays.

    Parameters
    ----------
    fun
        the function to be lowered
    in_axes
        Input axes. This can be a pytree of Sequences of Axis, where the pytree has the same structure as
        the args and each Sequence has length equal to the number of Axis or None (indicating an argument
        that should not be converted). Or it can be a pytree prefix, in which case everything below is
        treated with the same axes.
    out_axes
        The output axes

    Returns
    -------
    new fun
        keyword args are passed unchanged.
    """

    def wrapped(*numpy_args):
        def my_ntensor(in_axes, numpy_arg):
            if in_axes is None:
                return numpy_arg
            if not is_sequence_of_type(in_axes, (Axis, str)):
                raise ValueError(
                    f"lower found leaf node where {in_axes=} is not sequence of Axis/str")
            return ntensor(numpy_arg, *in_axes)

        # print(f"{fun=} {in_axes_pytree=} {out_axes_pytree=} {numpy_args=}")
        args = tree_map_axes(my_ntensor, in_axes_pytree, numpy_args)
        out = fun(*args)

        def my_array(out_axes, ntensor_out):
            return ntensor_out.numpy(*out_axes)

        return tree_map_axes(my_array, out_axes_pytree, out)

    return wrapped


def lower(fun: Callable, axes_str: str | None = None, in_axes=None, out_axes=None) -> Callable:
    in_axes, out_axes = parse_lift_str(axes_str, in_axes, out_axes)
    return lower_pytree(fun, in_axes, out_axes)


# Dot ##############################################################################################

def classify_axis_names(axes1: Set[str], axes2: Set[str]):
    shared = axes1 & axes2
    only1 = axes1 - shared
    only2 = axes2 - shared
    return shared, only1, only2


def n2c(num):
    """Converts an integer to its corresponding lowercase character."""
    if 0 <= num <= 25:
        return chr(num + 97)  # 97 is the ASCII value for 'a'
    else:
        raise Exception("Invalid input")


def count_axes(*arrays: ntensor):
    ax_counts = {}
    for ar in arrays:
        for ax in ar._axes_ordered:
            if ax not in ax_counts:
                ax_counts[ax] = 1
            else:
                ax_counts[ax] += 1
    return ax_counts


def get_remaining_axes(*arrays: ntensor, keep: abc.Set[str | Axis] = frozenset()):
    """Get the axes that will remain after doing einsum.
    Axes are ordered as encountered in the arrays.

    Examples
    --------
    >>> A = ntensor([[0,1,2],[3,4,5]],'i','j')
    >>> B = ntensor([[6,7,8],[9,10,11]],'j','k')
    >>> get_remaining_axes(A, B)
    (Axis('i'), Axis('k'))
    >>> get_remaining_axes(A, B, keep={'j'})
    (Axis('i'), Axis('j'), Axis('k'))
    """
    ax_counts = count_axes(*arrays)
    #print(f"{ax_counts=}")
    remaining_axes = []
    for ar in arrays:
        for ax in ar._axes_ordered:
            if ax not in remaining_axes:
                if ax_counts[ax] == 1 or ax in keep:
                    remaining_axes.append(ax)
    return tuple(remaining_axes)


def dot(*arrays: ntensor | Number, keep: Iterable[str | Axis] = ()) -> ntensor:
    """Do inner products or outer products or matrix multiplication or tensor multiplication or einstein
    summation or batched versions of any of these.

    Parameters
    ----------
    arrays
        The `NTensor` objects to operate on. In principle, the order does not matter. (Though it may affect
        the internal layout of the data.)
    keep
        Any shared indices you don't want reduced.

    Returns
    -------
    out
        Reduced `NTensor`

    Examples
    --------

    Inner product
    >>> a = ntensor([1,2,3],'i')
    >>> b = ntensor([10,0,100],'i')
    >>> dot(a,b)
    <ntensor {} 310>

    Elementwise product
    >>> dot(a,b,keep={'i'})
    <ntensor {i:3} [ 10   0 300]>

    Outer product
    >>> a = ntensor([0,10],'i')
    >>> b = ntensor([1,2,3],'j')
    >>> dot(a,b)
    <ntensor {i:2, j:3}
    [[ 0  0  0]
     [10 20 30]]>

    Matrix-vector multiplication
    >>> a = randn(i=5, j=4)
    >>> b = randn(j=4)
    >>> dot(a, b).shape
    ShapeDict(i=5)
    >>> dot(b, a).shape
    ShapeDict(i=5)
    >>> dot(a, b) == dot(b, a)
    <ntensor {i:5} [ True  True  True  True  True]>
    >>> c = randn(i=5)
    >>> dot(c, a).shape
    ShapeDict(j=4)
    >>> dot(c, a, b).shape
    ShapeDict()

    Matrix-matrix multiplication
    >>> a = randn(i=5, j=6)
    >>> b = randn(j=6, k=7)
    >>> dot(a, b).shape
    ShapeDict(i=5, k=7)

    Tensor multiplication / Einstein summation
    >>> a = randn(i=5, j=6, k=7)
    >>> b = randn(j=6, l=8)
    >>> c = randn(k=7, l=8, m=9)
    >>> dot(a, b, c).shape
    ShapeDict(i=5, m=9)
    >>> dot(a, b, c, keep={'j'}).shape
    ShapeDict(i=5, j=6, m=9)
    >>> dot(a, b, c, keep={'k','l'}).shape
    ShapeDict(i=5, k=7, l=8, m=9)
    """

    arrays = ensure_ntensor_tree(arrays)
    keep = set(keep)

    if not is_sequence_of_type(arrays, ntensor):
        raise ValueError("Arguments to do must be sequence of NTensors (got {arrays}).")

    all_axes = set().union(*(ar.axes for ar in arrays))
    remaining_axes = get_remaining_axes(*arrays, keep=keep)

    a2c = {}
    n = 0
    for axis in all_axes:
        a2c[axis] = n2c(n)
        n += 1

    array_chars = [[a2c[axis] for axis in ar._axes_ordered] for ar in arrays]

    # result should keep all non-shared dims, A first then B
    out_chars = [a2c[axis] for axis in remaining_axes]

    array_chars = [''.join(ar_chars) for ar_chars in array_chars]
    out_chars = ''.join(out_chars)

    subscripts = ""
    for ar_chars in array_chars:
        subscripts += ar_chars + ","
    subscripts = subscripts[:-1]
    subscripts += "->" + out_chars

    out_data = jnp.einsum(subscripts, *[A._data for A in arrays])
    return ntensor(out_data, *remaining_axes)


# vmap #############################################################################################

def prepare_args(args, in_axis):
    args_treedef = jax.tree.structure(args, is_leaf=is_ntensor)
    args_flat = jax.tree.leaves(args, is_leaf=is_ntensor)

    if not builtins.any(in_axis in arg.axes for arg in args_flat if isinstance(arg, ntensor)):
        raise ValueError(
            f"in axis {in_axis} doesn't appear in args with axes {[a.axes for a in args]}")

    numpy_in_axes_flat = []
    numpy_args_flat = []
    example_args_flat = []
    in_axes_flat = []
    for arg in args_flat:
        if not isinstance(arg, ntensor):
            my_example_arg = arg
            my_numpy_in_axis = None
            my_in_axes = None
            my_numpy_arg = arg
        elif in_axis not in arg.axes:
            my_example_arg = arg
            my_numpy_in_axis = None
            my_in_axes = arg._axes_ordered
            my_numpy_arg = arg.numpy(*my_in_axes)
        else:
            where = arg._axes_ordered.index(in_axis)
            ind = [slice(None)] * where + [0]
            my_example_arg_numpy = arg._data[*ind]
            my_in_axes = arg._axes_ordered[:where] + arg._axes_ordered[where + 1:]
            my_example_arg = ntensor(my_example_arg_numpy, *my_in_axes)
            my_numpy_in_axis = where
            my_numpy_arg = arg._data
        example_args_flat.append(my_example_arg)
        numpy_in_axes_flat.append(my_numpy_in_axis)
        in_axes_flat.append(my_in_axes)
        numpy_args_flat.append(my_numpy_arg)

    example_args = jax.tree.unflatten(args_treedef, example_args_flat)
    numpy_in_axes = jax.tree.unflatten(args_treedef, numpy_in_axes_flat)
    in_axes = jax.tree.unflatten(args_treedef, in_axes_flat)
    numpy_args = jax.tree.unflatten(args_treedef, numpy_args_flat)

    return example_args, numpy_args, in_axes, numpy_in_axes

def vmap_eval(fun: Callable, args: Tuple, in_axis: AxisLike):
    """
    Here's one very solid concept. You can define `vmap_eval(fun, [A_1, ..., A_N], [ax_1, ...,
    ax_N], out_ax)` and this will do the obvious thing provided and work provided that:
    1. `fun(A_1(ax_1=0), ..., A_N(ax_N=0))` works.
    2. `out_ax not in fun(A_1(ax_1=0), ..., A_N(ax_N=0)).axes`
    2. `A_1.shape[ax_1] == ... == A_N.shape[ax_N]`.

    The output will have shape `fun(A_1(ax_1=0), ..., A_N(ax_N=0)).shape | {out_ax: A_1.shape[
    ax_1]}`.  You can easily generalize this to allow `ax_n=None`. (At least one `ax_n` must be
    non-none.)

    Clearly in *principle*, all batched operations could be reduced to this.

    Examples
    --------

    >>> A = ntensor([0,1,2],'i')
    >>> B = ntensor([3,4,5],'i')
    >>> vmap_eval(add_scalar, (A, B), 'i')
    <ntensor {i:3} [3 5 7]>

    >>> A = ntensor([0,1,2],'i')
    >>> B = ntensor(3)
    >>> vmap_eval(add_scalar, (A, B), 'i')
    <ntensor {i:3} [3 4 5]>

    >>> A = ntensor([0,1,2],'i')
    >>> B = ntensor([3,4,5],'i')
    >>> myfun = lambda a, b: a+b*ntensor([0,10],'j')
    >>> vmap_eval(myfun, (A, B), 'i')
    <ntensor {i:3, j:2}
    [[ 0 30]
     [ 1 41]
     [ 2 52]]>
    """

    # TODO: tolerate non-ntensor inputs

    #args = ensure_ntensor_tree(args)
    in_axis = Axis(in_axis)

    #args_treedef = jax.tree.structure(args, is_leaf=is_ntensor)
    args_flat = jax.tree.leaves(args, is_leaf=is_ntensor)

    if not builtins.any(in_axis in arg.axes for arg in args_flat if isinstance(arg, ntensor)):
        raise ValueError(
            f"in axis {in_axis} doesn't appear in args with axes {[a.axes for a in args]}")

    example_args, numpy_args, in_axes, numpy_in_axes = prepare_args(args, in_axis)

    def convert1(a):
        if isinstance(a,ntensor):
            return a
        else:
            b = jnp.array(a)
            if b.ndim != 0:
                raise ValueError("Can't auto-convert data with > 0 dims to ntensor")
            return ntensor(b)

    def myfun(*args):
        "like fun except converts scalars to scalar ntensor"
        out = fun(*args)
        return jax.tree.map(convert1, out, is_leaf=is_ntensor)

    example_out = myfun(*example_args)

    if builtins.any([in_axis in e.axes for e in jax.tree.leaves(example_out, is_leaf=is_ntensor)]):
        raise ValueError(
            f"Can't vmap function over {in_axis=} with function that returns same mapped axis.")

    out_axes = jax.tree.map(lambda arg: arg._axes_ordered, example_out, is_leaf=is_ntensor)

    numpy_fun = lower_pytree(myfun, in_axes, out_axes)
    numpy_out = jax.vmap(numpy_fun, in_axes=numpy_in_axes)(*numpy_args)

    out = jax.tree.map(
        lambda numpy_o, example_o: ntensor(numpy_o, in_axis, *example_o._axes_ordered),
        numpy_out, example_out)
    return out


def vmap(fun: Callable, in_axes: AxisLike | Iterable[AxisLike] = frozenset(), other: None|Iterable[
    AxisLike]=None) -> (
        Callable):
    """Implements almost all of vmap. The only difference here is that for simplicity we require
    that in_axes and out_ax are provided in the wrapper call.

    Parameters
    ----------
    fun
        The function to be vmapped
    in_axes
        What input axis to map each argument over. If just given a single axis, all arguments that
        are `array`s and contain that axis are mapped over it and all other arguments are mapped
        over None.
    other
        iterable of other axes or None. If None (default), ignored. Otherwise, used as a check: all axes
        that appear in any argument must either appear in in_axes or other.

    Returns
    -------
    vfun
        vmapped function

    Examples
    --------

    >>> A = ntensor([0,1,2],'i')
    >>> B = ntensor([3,4,5],'i')
    >>> vmap(add_scalar, 'i')(A, B)
    <ntensor {i:3} [3 5 7]>

    >>> A = ntensor([0,1,2],'i')
    >>> B = ntensor(3)
    >>> vmap(add_scalar, 'i')(A, B)
    <ntensor {i:3} [3 4 5]>

    >>> A = ntensor([0,1,2],'i')
    >>> B = ntensor([10,20],'j')
    >>> vmap(add_scalar, ['i','j'])(A, B)
    <ntensor {i:3, j:2}
    [[10 20]
     [11 21]
     [12 22]]>

    >>> A = ntensor([0,1,2],'i')
    >>> B = ntensor([3,4,5],'i')
    >>> myfun = lambda a, b: a+b*ntensor([0,10],'j')
    >>> vmap(myfun, 'i')(A, B)
    <ntensor {i:3, j:2}
    [[ 0 30]
     [ 1 41]
     [ 2 52]]>

    """

    # if in_axes=={} (a dict), let's be forgiving
    if isinstance(in_axes, AxisLike):
        in_axes = (in_axes,)

    flat_in_axes = tuple(in_axes)

    #print(f"vmap called: {in_axes=}")

    def wrapped(*args):

        if other:
            remaining = other_axes(in_axes, *args)
            if set(remaining) != set(other):
                raise ValueError(f"Remaining axes {remaining} not equal to provided other={other}")

        if flat_in_axes == ():
            return fun(*args)
        elif len(in_axes) == 1:
            first = flat_in_axes[0]
            return vmap_eval(fun, args, first)
        else:
            first = flat_in_axes[-1]
            rest = set(flat_in_axes[:-1])
            return vmap(vmap(fun, first), rest)(*args)

    return wrapped

def batch(fun: Callable, axes: Iterable[AxisLike] = frozenset(), other:None|Iterable[AxisLike]=None) -> (
        Callable):
    """
    Create a batched version of a function

    Parameters
    ----------
    fun
        function to be batched
    axes: Set[Axis|Str]
        axes to be batched. Must be an `Axis` or str.
    other:
        optional: All other axes (for checking only)

    Returns
    -------
    new_fun
        batched function

    """
    axes = tuple(Axis(ax) for ax in axes)

    def wrapped(*args):
        #args = [convert_to_0dim_array_if_possible(arg) for arg in args]
        args = jax.tree.map(convert_to_0dim_array_if_possible, args, is_leaf=is_ntensor)

        vmap_axes = other_axes(axes, *args)

        return vmap(fun, vmap_axes, axes)(*args)


    return wrapped


def get_common_axes(args):
    common_axes = frozenset.intersection(*[arg.axes for arg in jax.tree.leaves(args, is_leaf=is_ntensor)])
    return tuple(ax for ax in args[0].axes if ax in common_axes) # preserve original order

def wrap(fun: Callable, kwargs=True) -> Callable:
    """Wrap a function so it can LATER be batched or vmapped by providing `axes` or `vmap_axes`
    arguments.

    `wrap(fun)(A, B, axes={i,j})` is equivalent to `batch(fun, {i,j})(A, B)`

    `wrap(fun)(A, B, vmap={k,l})` is equivalent to `vmap(fun, {k,l})(A, B)`

    `wrap(fun)(A, B, axes={i,j}, vmap={k,l})` is equivalent to `vmap(fun, {k,l}, other={i,j})(A, B)`

    """

    vmap_fun = vmap  # gets shadowed below

    if kwargs:
        def wrapped(*args, axes=None, vmap=None, **kwargs):
            args = ensure_ntensor_tree(args)
            axes = convert_empty_dict(axes)
            vmap = convert_empty_dict(vmap)

            fun_with_kwargs = lambda *args: fun(*args, **kwargs)

            if axes is None and vmap is None:
                return fun_with_kwargs(*args)
            elif vmap is not None:
                return vmap_fun(fun_with_kwargs, vmap, other=axes)(*args)
            else:
                assert axes is not None
                return batch(fun_with_kwargs, axes)(*args)
    else:
        def wrapped(*args, axes=None, vmap=None):
            args = ensure_ntensor_tree(args)
            axes = convert_empty_dict(axes)
            vmap = convert_empty_dict(vmap)

            if axes is None and vmap is None:
                return fun(*args)
            elif vmap is not None:
                return vmap_fun(fun, vmap, other=axes)(*args)
            else:
                assert axes is not None
                return batch(fun, axes)(*args)

    return wrapped


# Scan #########################################################################################

def scan(fun, init, xs, in_axis):
    in_axis = Axis(in_axis)

    # TODO: put in_axes in dimension 0 so scan can handle it
    # TODO: shield arguments that don't contain given dimension
    example_x, xs_numpy, x_in_axes, x_in_axes_numpy = prepare_args(xs, in_axis)

    example_carry, example_y = fun(init, example_x)

    out_axes = jax.tree.map(lambda arg: arg._axes_ordered, (example_carry, example_y), is_leaf=is_ntensor)

    if builtins.any(in_axis in i.axes for i in jax.tree.leaves(init, is_leaf=is_ntensor)):
        raise ValueError("Init can't contain scanned Axis")

    init_in_axes = jax.tree.map(lambda a: a._axes_ordered, init, is_leaf=is_ntensor)
    init_numpy = jax.tree.map(lambda a: a._data, init, is_leaf=is_ntensor)

    in_axes = (init_in_axes, x_in_axes)
    numpy_fun = lower_pytree(fun, in_axes, out_axes)
    numpy_carry, numpy_ys = jax.lax.scan(numpy_fun, init_numpy, xs_numpy)

    carry = jax.tree.map(
        lambda numpy_c, init_leaf: ntensor(numpy_c, *init_leaf._axes_ordered), numpy_carry, init)

    ys = jax.tree.map(
        lambda numpy_o, example_o: ntensor(numpy_o, in_axis, *example_o._axes_ordered),
        numpy_ys, example_y)

    return carry, ys

# Indexing #########################################################################################

def select_output_shape(input_shape: ShapeDict, axis_shapes: Dict[Axis, ShapeDict]) -> ShapeDict:
    """(Not intended for end usage.) Given an input shape and shapes for each axis, compute output
    shape. This function isn't intended for end usage, so it is very picky about inputs. But it
    might be useful for understanding what's happening under the hood.

    Parameters
    ----------
    input_shape: ShapeDict
        The shape of the array to be selected from
    axis_shapes: Dict[Axis, ShapeDict]
        The shapes for each Axis in the ShapeDict. All axes from input_shape must be included.
        (Notice that because it's a dict of dicts, repeated axes are impossible.)

    Returns
    -------
    output_shape: ShapeDict
        the shape of the output

    Examples
    --------

    The most basic usage is to resample a single dimension.

    >>> i,j,k,l = axes()
    >>> input_shape = ShapeDict(i=3)
    >>> input_shape
    ShapeDict(i=3)
    >>> axis_shapes = {i:ShapeDict(i=5)}
    >>> print(axis_shapes)
    {Axis('i'): ShapeDict(i=5)}
    >>> select_output_shape(input_shape, axis_shapes)
    ShapeDict(i=5)

    An input dimension can be sent to itself or to a new dimension.

    >>> select_output_shape(input_shape, {i:ShapeDict(i=5)})
    ShapeDict(i=5)
    >>> select_output_shape(input_shape, {i:ShapeDict(j=6)})
    ShapeDict(j=6)

    An input dimension can be expanded into multiple output dimensions.

    >>> select_output_shape(input_shape, {i:ShapeDict(j=4, k=5)})
    ShapeDict(j=4, k=5)

    You *can* include input dimensions in output dimensions.

    >>> select_output_shape(input_shape, {i:ShapeDict(j=3, i=5)})
    ShapeDict(j=3, i=5)

    You can collapse multiple axes together

    >>> input_shape = ShapeDict(i=3, j=4)
    >>> axis_shapes = {i:ShapeDict(k=3), j:ShapeDict(k=3)}
    >>> select_output_shape(input_shape, axis_shapes)
    ShapeDict(k=3)

    You can also do slightly crazy stuff.
    >>> input_shape = ShapeDict(i=3, j=4)
    >>> axis_shapes = {i:ShapeDict(k=5,i=6), j:ShapeDict(i=6,l=7)}
    >>> select_output_shape(input_shape, axis_shapes)
    ShapeDict(k=5, i=6, l=7)

    If you fail to give a shape for an input axis, you get an error.

    >>> input_shape = ShapeDict(i=3)
    >>> select_output_shape(input_shape, {})
    Traceback (most recent call last):
    ...
    ValueError: Input has axes (Axis('i'),) but only () given

    If you try to use inconsistent lengths for the same axis, you get an error.
    >>> i,j,k = axes()
    >>> input_shape = ShapeDict(i=3, j=4)
    >>> axis_shapes = {i:ShapeDict(k=3), j:ShapeDict(k=4)}
    >>> select_output_shape(input_shape, axis_shapes)
    Traceback (most recent call last):
    ...
    ValueError: Tried to assign k to 4 but already has value 3.


    """

    # dict(i=ShapeDict)

    if not builtins.all(ax in axis_shapes for ax in input_shape):
        raise ValueError(
            f"Input has axes {input_shape.axes} but only {tuple(axis_shapes.keys())} "
            f"given")

    output_shape = ConsistentDict()
    for axis in input_shape:
        shape = axis_shapes[axis]
        for ind_axis in shape:
            output_shape[ind_axis] = shape[ind_axis]

        # if axis not in axis_shapes:
        #     output_shape[axis] = input_shape[axis]
        # else:
        #     shape = axis_shapes[axis]
        #     for ind_axis in shape:
        #         output_shape[ind_axis] = shape[ind_axis]
    return ShapeDict(output_shape)


def select_pedantic(A: ntensor, axis_indices: Dict[Axis, ntensor]) -> ntensor:
    """(Not intended for end usage.) Given an input array and indices for each axis, compute output
    array. This function isn't intended for end usage, so it is very picky about inputs. But it
    might be useful for understanding what's happening under the hood.

    Parameters
    ----------
    A: ntensor
        to be selected from
    axis_indices: Dict[Axis,ntensor]
        dict mapping each input axis in array (must all be included) to an output axis

    Returns
    -------
    B: ntensor
        A after selecting

    Examples
    --------

    The simplest usage is to reindex a single dimension

    >>> i,j,k,l = axes()
    >>> A = ntensor([0, 10, 20, 30], i)
    >>> i_index = ntensor([2,0,1],i)
    >>> B = select_pedantic(A, {i:i_index})
    >>> B.shape
    ShapeDict(i=3)
    >>> print(B.numpy_broadcasted(i))
    [20  0 10]

    You can also expand that single dimension with a multi-dim index

    >>> A = ntensor([0, 10, 20, 30], i)
    >>> i_index = ntensor([[2,0,1],[4,4,4]],j,k)
    >>> B = select_pedantic(A, {i:i_index})
    >>> B.shape
    ShapeDict(j=2, k=3)
    >>> print(B.numpy_broadcasted(j,k))
    [[20  0 10]
     [30 30 30]]

    Here is how you do "pointwise" sampling from a 2d array:

    >>> A = ntensor([[0, 10, 20],[30, 40, 50]], i, j)
    >>> A.shape
    ShapeDict(i=2, j=3)
    >>> print(A.numpy_broadcasted(i,j))
    [[ 0 10 20]
     [30 40 50]]
    >>> i_index = ntensor([0,0,1,1], k)
    >>> j_index = ntensor([0,1,2,2], k)
    >>> B = select_pedantic(A, {i:i_index, j:j_index})
    >>> B.shape
    ShapeDict(k=4)
    >>> print(B.numpy_broadcasted(k))
    [ 0 10 50 50]

    Here is how you do "orthogonal" sampling from a 2d array:
    >>> A = ntensor([[0, 10, 20],[100, 110, 120]], i, j)
    >>> A.shape
    ShapeDict(i=2, j=3)
    >>> print(A.numpy_broadcasted(i,j))
    [[  0  10  20]
     [100 110 120]]
    >>> i_index = ntensor([1,0,1], i)
    >>> j_index = ntensor([2,0], j)
    >>> B = select_pedantic(A, {i:i_index, j:j_index})
    >>> B.shape
    ShapeDict(i=3, j=2)
    >>> print(B.numpy_broadcasted(i,j))
    [[120 100]
     [ 20   0]
     [120 100]]
    >>> # here's the horrorshow if you want to do that in numpy
    >>> i_index_numpy = jnp.array([[1],[0],[1]]) # 1x3
    >>> j_index_numpy = jnp.array([[2,0]]) # 2x1
    >>> B_numpy = A.numpy_broadcasted(i,j)[i_index_numpy, j_index_numpy]
    >>> print(B_numpy)
    [[120 100]
     [ 20   0]
     [120 100]]

    """

    if not builtins.all(isinstance(ax, Axis) for ax in axis_indices.keys()):
        raise TypeError("not all keys in axis_indices are of type Axis")
    if not builtins.all(isinstance(ar, ntensor) for ar in axis_indices.values()):
        raise TypeError("not all values in axis_indices are of type array")
    assert builtins.all(ax in axis_indices for ax in A.axes), "all axes must be indexed"
    axis_shapes = {ax: axis_indices[ax].shape for ax in axis_indices}
    output_shape = select_output_shape(A.shape, axis_shapes)
    output_axes = tuple(output_shape.keys())
    # go through each INPUT axis and put indices into shape that will broadcast to output size
    numpy_indices = []
    for in_ax in A._axes_ordered:
        unordered_indices = axis_indices[in_ax]
        # put in order and add blank dims for broadcasting
        my_numpy_indices = unordered_indices.numpy_broadcasted(*output_axes)
        numpy_indices.append(my_numpy_indices)

    return ntensor(A._data[*numpy_indices], *output_axes)


def select(A: ntensor, **axis_indices: AxisIndices) -> ntensor:
    """Select/index data. Note you can also using indexing. One of the core highlights of this
    package is how general and simple this is.

    Parameters
    ----------
    A: ntensor
        the array to select/index from
    **axis_indices
        mapping from str or Axis to int or slice or Sequence or 1-d array

    Returns
    -------
    B: ntensor
        NTensor after indexing

    Examples
    --------

    Basic indexing of 1-d arrays with integers, lists, slices

    >>> A = ntensor([0,10,20,30,40,50,60,70,80],'n')
    >>> print(A.shape)
    {n:9}
    >>> B = select(A,n=4)
    >>> print(B.shape)
    {}
    >>> print(B.numpy_broadcasted())
    40
    >>> B = select(A,n=[2,0,3])
    >>> print(B.shape)
    {n:3}
    >>> print(B.numpy_broadcasted('n'))
    [20  0 30]
    >>> C = select(A,n=slice(1,5))
    >>> print(C.shape)
    {n:4}
    >>> print(C.numpy_broadcasted('n'))
    [10 20 30 40]
    >>> D = select(A,n=slice(0,8,2))
    >>> print(D.shape)
    {n:4}
    >>> print(D.numpy_broadcasted('n'))
    [ 0 20 40 60]

    More advanced indexing

    >>> A = ones(i=5, j=6, k=8, l=9) # complex example that shows almost all functionality
    >>> B = select(A, i=0, j=slice(None,None,2), k=[5,4,1,1])
    >>> print(B.shape)
    {j:3, k:4, l:9}
    >>> B = select(A, i=0, j=slice(None,None,2), k=ntensor([5,4,1,1],'m'))
    >>> print(B.shape)
    {j:3, m:4, l:9}

    See Also
    --------
    select_pedantic
    select_output_shape
    """
    # TODO: avoid converting slices into arrays

    # if args:
    #     extra_kwargs = {}
    #     for arg in args:
    #         assert isinstance(arg, IndexedAxis)
    #         extra_kwargs[arg.axis.name] = arg
    #
    #     return select(A, **kwargs, **extra_kwargs)

    # force all keys to be of type Axis
    axis_indices = {Axis(ax): axis_indices[ax] for ax in axis_indices}

    # create full slices for any indices that aren't mentioned
    for ax in A.axes:
        if ax not in axis_indices:
            axis_indices[ax] = slice(None)

    new_kwargs = {}
    for ax in axis_indices:
        if ax not in A.axes:
            raise ValueError(f"Array with shape {A.shape} indexed with nonexistent axis {ax}.")

        ind = axis_indices[ax]

        # # allow usage like i=j rather than i=j[:] ?
        # if isinstance(ind, Axis):
        #     ind = IndexedAxis(ind, slice(None))

        if isinstance(ind, IndexedAxis):
            if isinstance(ind.index, slice):
                # for slices, size is determined by the INPUT axes
                if ax in A.shape:
                    ax_len = A.shape[ax]
                elif ind.index.stop is not None:
                    ax_len = ind.index.stop
                else:
                    raise ValueError(
                        "If using NEW index as coord with slice, stop must be provided")

                new_kwargs[ax] = ntensor(jnp.arange(ax_len)[ind.index], ind.axis)
            else:
                data = jnp.array(ind)
                if data.ndim != 1:
                    raise ValueError(f"Must have 1 dims (got {data.ndim})")
                new_kwargs[ax] = ntensor(data, ind.axis)
        elif isinstance(ind, ntensor):
            new_kwargs[ax] = axis_indices[ax]
        elif isinstance(ind, slice):
            new_kwargs[ax] = ntensor(jnp.arange(A.shape[ax])[ind], ax)
        else:
            data = jnp.array(ind)
            if data.ndim == 0:
                new_kwargs[ax] = ntensor(data)
            elif data.ndim == 1:
                new_kwargs[ax] = ntensor(data, ax)
            else:
                raise ValueError(f"can't provide non-NTensor index with N>1 dims (got {ax} index with shape"
                                 f"={data.shape})")
    return select_pedantic(A, new_kwargs)


# Scalar ops #######################################################################################

add_scalar = lift_pytree(jnp.add)
multiply_scalar = lift_pytree(jnp.multiply)
subtract_scalar = lift_pytree(jnp.subtract)
divide_scalar = lift_pytree(jnp.divide)
pow_scalar = lift_pytree(jnp.pow)


def wrap_jax_elementwise(jax_fun: Callable) -> Callable:
    # def new_fun(arg: ntensor):
    #     assert isinstance(arg, ntensor)
    #     return ntensor(jax_fun(arg._data), *arg._axes_ordered)

    #return batch_wrap(new_fun)

    return lift(jax_fun)


abs = wrap_jax_elementwise(jnp.abs)
"Elementwise operation (see same operation in [`jax.numpy`](https://docs.jax.dev/en/latest/jax.numpy.html))"
acos = wrap_jax_elementwise(jnp.acos)
"Elementwise operation (see same operation in [`jax.numpy`](https://docs.jax.dev/en/latest/jax.numpy.html))"
acosh = wrap_jax_elementwise(jnp.acosh)
"Elementwise operation (see same operation in [`jax.numpy`](https://docs.jax.dev/en/latest/jax.numpy.html))"
arccos = wrap_jax_elementwise(jnp.arccos)
"Elementwise operation (see same operation in [`jax.numpy`](https://docs.jax.dev/en/latest/jax.numpy.html))"
arccosh = wrap_jax_elementwise(jnp.arccosh)
"Elementwise operation (see same operation in [`jax.numpy`](https://docs.jax.dev/en/latest/jax.numpy.html))"
arcsin = wrap_jax_elementwise(jnp.arcsin)
"Elementwise operation (see same operation in [`jax.numpy`](https://docs.jax.dev/en/latest/jax.numpy.html))"
arcsinh = wrap_jax_elementwise(jnp.arcsinh)
"Elementwise operation (see same operation in [`jax.numpy`](https://docs.jax.dev/en/latest/jax.numpy.html))"
arctan = wrap_jax_elementwise(jnp.arctan)
"Elementwise operation (see same operation in [`jax.numpy`](https://docs.jax.dev/en/latest/jax.numpy.html))"
arctanh = wrap_jax_elementwise(jnp.arctanh)
"Elementwise operation (see same operation in [`jax.numpy`](https://docs.jax.dev/en/latest/jax.numpy.html))"
asin = wrap_jax_elementwise(jnp.asin)
"Elementwise operation (see same operation in [`jax.numpy`](https://docs.jax.dev/en/latest/jax.numpy.html))"
asinh = wrap_jax_elementwise(jnp.asinh)
"Elementwise operation (see same operation in [`jax.numpy`](https://docs.jax.dev/en/latest/jax.numpy.html))"
atan = wrap_jax_elementwise(jnp.atan)
"Elementwise operation (see same operation in [`jax.numpy`](https://docs.jax.dev/en/latest/jax.numpy.html))"
atanh = wrap_jax_elementwise(jnp.atanh)
"Elementwise operation (see same operation in [`jax.numpy`](https://docs.jax.dev/en/latest/jax.numpy.html))"
ceil = wrap_jax_elementwise(jnp.ceil)
"Elementwise operation (see same operation in [`jax.numpy`](https://docs.jax.dev/en/latest/jax.numpy.html))"
conj = wrap_jax_elementwise(jnp.conj)
"Elementwise operation (see same operation in [`jax.numpy`](https://docs.jax.dev/en/latest/jax.numpy.html))"
cos = wrap_jax_elementwise(jnp.cos)
"Elementwise operation (see same operation in [`jax.numpy`](https://docs.jax.dev/en/latest/jax.numpy.html))"
cosh = wrap_jax_elementwise(jnp.cosh)
"Elementwise operation (see same operation in [`jax.numpy`](https://docs.jax.dev/en/latest/jax.numpy.html))"
deg2rad = wrap_jax_elementwise(jnp.deg2rad)
"Elementwise operation (see same operation in [`jax.numpy`](https://docs.jax.dev/en/latest/jax.numpy.html))"
exp = wrap_jax_elementwise(jnp.exp)
"Elementwise operation (see same operation in [`jax.numpy`](https://docs.jax.dev/en/latest/jax.numpy.html))"
exp2 = wrap_jax_elementwise(jnp.exp2)
"Elementwise operation (see same operation in [`jax.numpy`](https://docs.jax.dev/en/latest/jax.numpy.html))"
expm1 = wrap_jax_elementwise(jnp.expm1)
"Elementwise operation (see same operation in [`jax.numpy`](https://docs.jax.dev/en/latest/jax.numpy.html))"
i0 = wrap_jax_elementwise(jnp.i0)
"Elementwise operation (see same operation in [`jax.numpy`](https://docs.jax.dev/en/latest/jax.numpy.html))"
imag = wrap_jax_elementwise(jnp.imag)
"Elementwise operation (see same operation in [`jax.numpy`](https://docs.jax.dev/en/latest/jax.numpy.html))"
log = wrap_jax_elementwise(jnp.log)
"Elementwise operation (see same operation in [`jax.numpy`](https://docs.jax.dev/en/latest/jax.numpy.html))"
log10 = wrap_jax_elementwise(jnp.log10)
"Elementwise operation (see same operation in [`jax.numpy`](https://docs.jax.dev/en/latest/jax.numpy.html))"
log1p = wrap_jax_elementwise(jnp.log1p)
"Elementwise operation (see same operation in [`jax.numpy`](https://docs.jax.dev/en/latest/jax.numpy.html))"
log2 = wrap_jax_elementwise(jnp.log2)
"Elementwise operation (see same operation in [`jax.numpy`](https://docs.jax.dev/en/latest/jax.numpy.html))"
rad2deg = wrap_jax_elementwise(jnp.rad2deg)
"Elementwise operation (see same operation in [`jax.numpy`](https://docs.jax.dev/en/latest/jax.numpy.html))"
real = wrap_jax_elementwise(jnp.real)
"Elementwise operation (see same operation in [`jax.numpy`](https://docs.jax.dev/en/latest/jax.numpy.html))"
reciprocal = wrap_jax_elementwise(jnp.reciprocal)
"Elementwise operation (see same operation in [`jax.numpy`](https://docs.jax.dev/en/latest/jax.numpy.html))"
rint = wrap_jax_elementwise(jnp.rint)
"Elementwise operation (see same operation in [`jax.numpy`](https://docs.jax.dev/en/latest/jax.numpy.html))"
round = wrap_jax_elementwise(jnp.round)
"Elementwise operation (see same operation in [`jax.numpy`](https://docs.jax.dev/en/latest/jax.numpy.html))"
sign = wrap_jax_elementwise(jnp.sign)
"Elementwise operation (see same operation in [`jax.numpy`](https://docs.jax.dev/en/latest/jax.numpy.html))"
sin = wrap_jax_elementwise(jnp.sin)
"Elementwise operation (see same operation in [`jax.numpy`](https://docs.jax.dev/en/latest/jax.numpy.html))"
sinc = wrap_jax_elementwise(jnp.sinc)
"Elementwise operation (see same operation in [`jax.numpy`](https://docs.jax.dev/en/latest/jax.numpy.html))"
sinh = wrap_jax_elementwise(jnp.sinh)
"Elementwise operation (see same operation in [`jax.numpy`](https://docs.jax.dev/en/latest/jax.numpy.html))"
sqrt = wrap_jax_elementwise(jnp.sqrt)
"Elementwise operation (see same operation in [`jax.numpy`](https://docs.jax.dev/en/latest/jax.numpy.html))"
square = wrap_jax_elementwise(jnp.square)
"Elementwise operation (see same operation in [`jax.numpy`](https://docs.jax.dev/en/latest/jax.numpy.html))"
tan = wrap_jax_elementwise(jnp.tan)
"Elementwise operation (see same operation in [`jax.numpy`](https://docs.jax.dev/en/latest/jax.numpy.html))"
tanh = wrap_jax_elementwise(jnp.tanh)
"Elementwise operation (see same operation in [`jax.numpy`](https://docs.jax.dev/en/latest/jax.numpy.html))"
trunc = wrap_jax_elementwise(jnp.trunc)
"Elementwise operation (see same operation in [`jax.numpy`](https://docs.jax.dev/en/latest/jax.numpy.html))"
relu = wrap_jax_elementwise(jax.nn.relu)
"Elementwise operation (see same operation in [`jax.nn`](https://docs.jax.dev/en/latest/jax.nn.html))"


# Pairwise functions ###############################################################################

wrap_jax_pairwise = lift

add = wrap_jax_pairwise(jnp.add)
"""Pairwise operation defined simply as `wrap_jax_pairwise(jnp.add)`
(see same operation in [`jax.numpy`](https://docs.jax.dev/en/latest/jax.numpy.html))
"""
arctan2 = wrap_jax_elementwise(jnp.arctan2)
"Pairwise operation (see same operation in [`jax.numpy`](https://docs.jax.dev/en/latest/jax.numpy.html))"
atan2 = wrap_jax_elementwise(jnp.arctan2)
"Pairwise operation (see same operation in [`jax.numpy`](https://docs.jax.dev/en/latest/jax.numpy.html))"
divide = wrap_jax_pairwise(jnp.divide)
"Pairwise operation (see same operation in [`jax.numpy`](https://docs.jax.dev/en/latest/jax.numpy.html))"
divmod = wrap_jax_pairwise(jnp.divmod)
"Pairwise operation (see same operation in [`jax.numpy`](https://docs.jax.dev/en/latest/jax.numpy.html))"
equal = wrap_jax_pairwise(jnp.equal)
"Pairwise operation (see same operation in [`jax.numpy`](https://docs.jax.dev/en/latest/jax.numpy.html))"
floor_divide = wrap_jax_pairwise(jnp.floor_divide)
"Pairwise operation (see same operation in [`jax.numpy`](https://docs.jax.dev/en/latest/jax.numpy.html))"
gcd = wrap_jax_pairwise(jnp.gcd)
"Pairwise operation (see same operation in [`jax.numpy`](https://docs.jax.dev/en/latest/jax.numpy.html))"
greater = wrap_jax_pairwise(jnp.greater)
"Pairwise operation (see same operation in [`jax.numpy`](https://docs.jax.dev/en/latest/jax.numpy.html))"
greater_equal = wrap_jax_pairwise(jnp.greater_equal)
"Pairwise operation (see same operation in [`jax.numpy`](https://docs.jax.dev/en/latest/jax.numpy.html))"
heaviside = wrap_jax_pairwise(jnp.heaviside)
"Pairwise operation (see same operation in [`jax.numpy`](https://docs.jax.dev/en/latest/jax.numpy.html))"
hypot = wrap_jax_pairwise(jnp.hypot)
"Pairwise operation (see same operation in [`jax.numpy`](https://docs.jax.dev/en/latest/jax.numpy.html))"
lcm = wrap_jax_pairwise(jnp.lcm)
"Pairwise operation (see same operation in [`jax.numpy`](https://docs.jax.dev/en/latest/jax.numpy.html))"
ldexp = wrap_jax_pairwise(jnp.ldexp)
"Pairwise operation (see same operation in [`jax.numpy`](https://docs.jax.dev/en/latest/jax.numpy.html))"
less = wrap_jax_pairwise(jnp.less)
"Pairwise operation (see same operation in [`jax.numpy`](https://docs.jax.dev/en/latest/jax.numpy.html))"
less_equal = wrap_jax_pairwise(jnp.less_equal)
"Pairwise operation (see same operation in [`jax.numpy`](https://docs.jax.dev/en/latest/jax.numpy.html))"
logaddexp = wrap_jax_pairwise(jnp.logaddexp)
"Pairwise operation (see same operation in [`jax.numpy`](https://docs.jax.dev/en/latest/jax.numpy.html))"
logaddexp2 = wrap_jax_pairwise(jnp.logaddexp2)
"Pairwise operation (see same operation in [`jax.numpy`](https://docs.jax.dev/en/latest/jax.numpy.html))"
maximum = wrap_jax_pairwise(jnp.maximum)
"Pairwise operation (see same operation in [`jax.numpy`](https://docs.jax.dev/en/latest/jax.numpy.html))"
minimum = wrap_jax_pairwise(jnp.minimum)
"Pairwise operation (see same operation in [`jax.numpy`](https://docs.jax.dev/en/latest/jax.numpy.html))"
multiply = wrap_jax_pairwise(jnp.multiply)
"Pairwise operation (see same operation in [`jax.numpy`](https://docs.jax.dev/en/latest/jax.numpy.html))"
mod = wrap_jax_pairwise(jnp.mod)
"Pairwise operation (see same operation in [`jax.numpy`](https://docs.jax.dev/en/latest/jax.numpy.html))"
nextafter = wrap_jax_pairwise(jnp.nextafter)
"Pairwise operation (see same operation in [`jax.numpy`](https://docs.jax.dev/en/latest/jax.numpy.html))"
not_equal = wrap_jax_pairwise(jnp.not_equal)
"Pairwise operation (see same operation in [`jax.numpy`](https://docs.jax.dev/en/latest/jax.numpy.html))"
polyadd = wrap_jax_pairwise(jnp.polyadd)
"Pairwise operation (see same operation in [`jax.numpy`](https://docs.jax.dev/en/latest/jax.numpy.html))"
polymul = wrap_jax_pairwise(jnp.polymul)
"Pairwise operation (see same operation in [`jax.numpy`](https://docs.jax.dev/en/latest/jax.numpy.html))"
polysub = wrap_jax_pairwise(jnp.polysub)
"Pairwise operation (see same operation in [`jax.numpy`](https://docs.jax.dev/en/latest/jax.numpy.html))"
polyval = wrap_jax_pairwise(jnp.polyval)
"Pairwise operation (see same operation in [`jax.numpy`](https://docs.jax.dev/en/latest/jax.numpy.html))"
pow = wrap_jax_pairwise(jnp.pow)
"Pairwise operation (see same operation in [`jax.numpy`](https://docs.jax.dev/en/latest/jax.numpy.html))"
power = wrap_jax_pairwise(jnp.pow)
"Pairwise operation (see same operation in [`jax.numpy`](https://docs.jax.dev/en/latest/jax.numpy.html))"
remainder = wrap_jax_pairwise(jnp.remainder)
"Pairwise operation (see same operation in [`jax.numpy`](https://docs.jax.dev/en/latest/jax.numpy.html))"
subtract = wrap_jax_pairwise(jnp.subtract)
"Pairwise operation (see same operation in [`jax.numpy`](https://docs.jax.dev/en/latest/jax.numpy.html))"
true_divide = wrap_jax_pairwise(jnp.true_divide)
"Pairwise operation (see same operation in [`jax.numpy`](https://docs.jax.dev/en/latest/jax.numpy.html))"


# Reductions #######################################################################################

def wrap_jax_reduction(jax_fun):
    def new_fun(arg: ntensor):
        assert isinstance(arg, ntensor)
        return ntensor(jax_fun(arg._data))

    return wrap(new_fun, kwargs=False)

any = wrap_jax_reduction(jnp.any)
"""Wraps [`jax.numpy.any`](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.any.html). Call 
with `axes` to reduce or `vmap_axes` to *not* reduce."""
all = wrap_jax_reduction(jnp.all)
"""Wraps [`jax.numpy.all`](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.all.html). 
Call with `axes` to reduce or `vmap_axes` to *not* reduce."""
logsumexp = wrap_jax_reduction(jax.nn.logsumexp)
"""Wraps [`jax.nn.logsumexp`](https://docs.jax.dev/en/latest/_autosummary/jax.nn.logsumexp.html). 
Call with `axes` to reduce or `vmap_axes` to *not* reduce."""
max = wrap_jax_reduction(jnp.max)
"""Wraps [`jax.numpy.max`](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.max.html). Call 
with `axes` to reduce or `vmap_axes` to *not* reduce."""
mean = wrap_jax_reduction(jnp.mean)
"""Wraps [`jax.numpy.mean`](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.mean.html). 
Call with `axes` to reduce or `vmap_axes` to *not* reduce."""
median = wrap_jax_reduction(jnp.median)
"""Wraps [`jax.numpy.median`](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.median.html). 
Call with `axes` to reduce or `vmap_axes` to *not* reduce."""
min = wrap_jax_reduction(jnp.min)
"""Wraps [`jax.numpy.min`](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.min.html). Call 
with `axes` to reduce or `vmap_axes` to *not* reduce."""
prod = wrap_jax_reduction(jnp.prod)
"""Wraps [`jax.numpy.prod`](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.prod.html). 
Call with `axes` to reduce or `vmap_axes` to *not* reduce."""
sum = wrap_jax_reduction(jnp.sum)
"""Wraps [`jax.numpy.sum`](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.sum.html). Call 
with `axes` to reduce or `vmap_axes` to *not* reduce."""
std = wrap_jax_reduction(jnp.std)
"""Wraps [`jax.numpy.std`](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.std.html). 
Call with `axes` to reduce or `vmap_axes` to *not* reduce."""
var = wrap_jax_reduction(jnp.var)
"""Wraps [`jax.numpy.var`](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.var.html). 
Call with `axes` to reduce or `vmap_axes` to *not* reduce."""


# Gradients ########################################################################################

def grad(fun, argnums=0):
    """A thin wrapper around [`jax.grad`](https://docs.jax.dev/en/latest/_autosummary/jax.grad.html). All
    this really does is convert the final `NTensor` output to a jax value.

    Parameters
    ----------
    fun
        The function you will take the gradient of
    argnums
        Optional, integer or sequence of integers, specifying what arguments to take gradient with respect
        to. Defaults to 0.

    Returns
    -------
    grad_fun
        Gradient function

    Examples
    --------
    >>> x = ntensor([0., 1., 2.],'i')
    >>> grad_sum = grad(sum)
    >>> grad_sum(x)
    <ntensor {i:3} [1. 1. 1.]>

    You can use regular jax grad if you really want to (this is effectively all this function does).

    >>> x = ntensor([0., 1., 2.],'i')
    >>> mysum = lambda arg: sum(arg).numpy() # converts to jax float
    >>> sum(x)
    <ntensor {} 3.0>
    >>> mysum(x)
    Array(3., dtype=float32)
    >>> grad_sum = jax.grad(mysum)
    >>> grad_sum(x)
    <ntensor {i:3} [1. 1. 1.]>

    """

    def myfun(*args, **kwargs):
        out = fun(*args, **kwargs)
        if not (isinstance(out, ntensor) and out.shape == {}):
            raise ValueError(f"Gradient can only be taken of scalar functions (got {out=}).")
        return out._data

    mygrad = jax.grad(myfun, argnums=argnums)

    def wrapped_grad(*args, **kwargs):
        args = ensure_ntensor_tree(args)
        return mygrad(*args, **kwargs)

    return wrapped_grad


def value_and_grad(fun, argnums=0):
    """A thin wrapper around
    [`jax.value_and_grad`](https://docs.jax.dev/en/latest/_autosummary/jax.value_and_grad.html).


    Parameters
    ----------
    fun
        The function you will take the gradient of
    argnums
        Optional, integer or sequence of integers, specifying what arguments to take gradient with respect
        to. Defaults to 0.

    Returns
    -------
    value_and_grad_fun
        Function that returns the value and the gradient

    Examples
    --------
    >>> x = ntensor([0., 1., 2.],'i')
    >>> value_and_grad_sum = value_and_grad(sum)
    >>> value_and_grad_sum(x)
    (<ntensor {} 3.0>, <ntensor {i:3} [1. 1. 1.]>)
    """

    def myfun(*args, **kwargs):
        out = fun(*args, **kwargs)
        if not (isinstance(out, ntensor) and out.shape == {}):
            raise ValueError(f"Gradient can only be taken of scalar functions (got {out=}).")
        return out._data

    my_value_and_grad = jax.value_and_grad(myfun, argnums=argnums)

    def wrapped_value_and_grad(*args, **kwargs):
        args = ensure_ntensor_tree(args)
        out = my_value_and_grad(*args, **kwargs)
        return ntensor(out[0]), out[1]

    return wrapped_value_and_grad


# Other matrix operations ##########################################################################

@wrap
def solve(a: ntensor, b: ntensor) -> ntensor:
    if a.ndim != 2:
        raise ValueError(f"first argument to solve (after batching) must have 2 dims (got {a.ndim}")
    if b.ndim != 1:
        raise ValueError(f"second argument to solve (after batching) must have 1 dim (got {b.ndim}")
    if len(a.axes & b.axes) != 1:
        raise ValueError(f"arguments must overlap on exactly one axis (got {a.axes | b.axes})")

    new_axis = a.axes - b.axes
    assert len(new_axis) == 1
    new_axis = list(new_axis)[0]

    c_jax = jnp.linalg.solve(a._data, b._data)
    return ntensor(c_jax, new_axis)


@wrap
def inv(a: ntensor) -> ntensor:
    """Take an inverse

    Parameters
    ----------
    a
        2-D NTensor

    Returns
    -------
    b
        Inverse NTensor

    Examples
    --------
    >>> a = ntensor([[2., 0],[0, 1]],'i','j')
    >>> inv(a)
    <ntensor {j:2 i:2}
    [[0.5 0. ]
     [0.  1. ]]>
    >>> b = ntensor([5., 3.],'i')
    >>> x = inv(a) @ b
    >>> x
    <ntensor {j:2} [2.5 3. ]>
    >>> a @ x
    <ntensor {i:2} [5. 3.]>
    """

    if a.ndim != 2:
        raise ValueError(f"argument to inv (after batching) must have 2 dims (got {a.ndim}")

    b_jax = jnp.linalg.inv(a._data)
    return ntensor(b_jax, *reversed(a._axes_ordered))

def conv_1d(a: ntensor, b: ntensor) -> ntensor:
    # axis = Axis(Axis)

    # if a.axes != {axis} or b.axes != {axis}:
    #    raise ValueError(f"conv_1d get {a.shape=} {b.shape=} not matching {axis=}")

    # print(f"{a.shape=} {b.shape=}")

    if a.ndim != 1:
        raise ValueError("Convolve must take 1d arg")
    if b.ndim != 1:
        raise ValueError("Convolve must take 1d arg")
    if len(a.axes & b.axes) != 1:
        raise ValueError("Convolve args must share dimension")

    [ax] = a._axes_ordered
    return lift(jnp.convolve, in_axes=([ax], [ax]), out_axes=[ax])(a, b)


#

@wrap
def convolve(a: ntensor, b: ntensor, mode='full') -> ntensor:
    if a.axes != b.axes:
        raise ValueError("Can't convolve with non-matching axes")

    axes = a._axes_ordered

    return ntensor(jax.scipy.signal.convolve(a.numpy(*axes), b.numpy(*axes), mode=mode), *axes)

# Dataframe ##################################################################################################

def dataframe(A: ntensor, label:str, *axis_arrays):
    """Convert ntensor to dataframe

    Parameters
    ----------
    A
    label
    axis_arrays

    Returns
    -------
    out: DataFrame
        Dataframe representing all the data in A, in "tidy" format

    Examples
    --------
    >>> A = ntensor([10, 25.5, 30], 'dogs')
    >>> dog_indices = ntensor([2, 5, 10],'dogs')
    >>> df = dataframe(A, 'weight', dog_indices)
    >>> df
       weight  dogs
    0    10.0     2
    1    25.5     5
    2    30.0    10

    >>> A = ntensor([[10, 25.5, 30],[44.5, 50.0, 0.0]], 'dogs', 'meals')
    >>> dog_indices = ntensor([2, 5],'dogs')
    >>> meal_indices = ntensor([3.3, 1.1, 10],'meals')
    >>> dataframe(A, 'weight', dog_indices, meal_indices)
       weight  dogs  meals
    0    10.0     2    3.3
    1    25.5     2    1.1
    2    30.0     2   10.0
    3    44.5     5    3.3
    4    50.0     5    1.1
    5     0.0     5   10.0
    >>> dataframe(A, 'weight', meal_indices, dog_indices)
       weight  meals  dogs
    0    10.0    3.3     2
    1    44.5    3.3     5
    2    25.5    1.1     2
    3    50.0    1.1     5
    4    30.0   10.0     2
    5     0.0   10.0     5
    """

    import pandas as pd

    axes = []
    for axis_array in axis_arrays:
        assert axis_array.ndim == 1
        axis, = axis_array.axes
        assert axis not in axes
        axes.append(axis)

    assert set(axes) == A.axes

    # broadcast axis_arrays to full size
    axis_arrays = [axis_array*ones_like(A, dtype=axis_array.dtype) for axis_array in axis_arrays]

    data = [A.numpy(*axes).ravel()] + [axis_array.numpy(*axes).ravel() for axis_array in axis_arrays]
    columns = [label] + [axis._name for axis in axes]
    df = pd.DataFrame({col:d for col, d in zip(columns, data)})
    return df


