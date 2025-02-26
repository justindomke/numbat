# Numbat

NumPy+Jax with named axes and an uncompromising attitude

![numbat](numbat.jpg)

## Manifesto

Does this resonate with you? 

* In NumPy (and PyTorch and Jax et al.), broadcasting and batching and indexing are confusing and tedious.

* Einstein summation, meanwhile, is good.

* But why only Einstein *summation*? Why not Einstein *everything*?

* And why not have the arrays remember which axis goes where, so you don't have to keep repeating that?

If so, you might like this package.

## Table of Contents

<!-- TOC -->
* [Numbat](#numbat)
  * [Manifesto](#manifesto)
  * [Table of Contents](#table-of-contents)
  * [Requirements](#requirements)
  * [Installation](#installation)
  * [Tutorials](#tutorials)
  * [Why use this instead of Jax?](#why-use-this-instead-of-jax)
    * [Example 1: Indexing](#example-1-indexing-)
    * [Example 2: Batching](#example-2-batching)
  * [Minimal API](#minimal-api)
  * [Full API docs](#full-api-docs)
  * [Using Jax features](#using-jax-features)
    * [Things that work out of the box](#things-that-work-out-of-the-box)
    * [Gradient functions](#gradient-functions)
    * [Things that don't work](#things-that-dont-work)
  * [The sharp bits](#the-sharp-bits)
  * [How broadcasting works](#how-broadcasting-works)
  * [Friends](#friends)
<!-- TOC -->

## Requirements

* Python 3.10+
* Numpy
* Jax
* [varname](https://github.com/pwwang/python-varname) (Optional: For magical axis naming.)
* Pandas (Optional: If you want to use `dataframe`)

## Installation

1. It's a single file: [`numbat.py`](https://github.com/justindomke/numbat/blob/main/numbat.py)
2. Download it and put it in your directory.
3. Done.

## Tutorials

* [Convnet tutorial](https://github.com/justindomke/numbat/blob/main/convnet_tutorial.ipynb)
* [Minimal API tutorial](https://github.com/justindomke/numbat/blob/main/minimal_api_tutorial.ipynb)
* [Maximal API tutorial](https://github.com/justindomke/numbat/blob/main/maximal_api_tutorial.ipynb)

## Why use this instead of Jax?

First of all, you don't have to use it *instead*, you can use them together. Numbat is a different interface—all the real work is still done by Jax. You can start by using Numbat *inside* your existing Jax code, in whatever spots that makes things easier. All the standard Jax features still work (GPUs, JIT compilation, gradients, etc.) and interoperate smoothly.

OK, but *when* would Numbat make things easier?  Well, in NumPy (and Jax and PyTorch), easy things are already easy, and Numbat will not help. But hard things are often *really* hard, because:

* Indexing gets insanely complicated and tedious.
* Broadcasting gets insanely complicated and tedious.
* Writing "batched" code gets insanely complicated and tedious.

Ultimately, these all stem from the same issue: Numpy indexes different axes by *position*. This leads to constant, endless fiddling to get the axes of different arrays to align with each other. It also means that different library functions all have their own (varying, and often poorly documented) conventions on where the different axes are supposed to go and what happens when arrays of different numbers of dimensions are provided.

Numbat is an experiment. What if axes didn't *have* positions, but only *names*? Sure, the bits have to be laid out in some order, but why make the user think about that? Following [many previous projects](#friends), let's define the shape to be a *dictionary* that maps names to ints. But what if we're totally uncompromising and *only* allow indexing using names? And what if we redesign indexing and broadcasting and batching around that representation? Does something nice happen?

This is still just a prototype. But I think it's enough to validate that the answer is yes: Something very nice happens.

### Example 1: Indexing 

Say you've got some array `X` containing data from different users, at different times and with different features. You've got a few different subsets of users stored in `my_users`. For each user, there is some subset of times you care about, stored in `my_times`. For each user/subset/time combination, there's one feature you care about, stored in an array `my_feats`. So this is your situation:

```
X.shape        == (n_user, n_time, n_feat)
my_users.shape == (100, 5)
my_times.shape == (20, 100)
my_feats.shape == (20, 5, 100)
```

You want to produce an array `Z` such that for all combinations of `i`, `j`, and `k`, the following is true:

```
Z[i,j,k] == X[my_users[i,k], my_times[j,i], my_feats[j,k,i]]
```

What's the easiest way to do that in NumPy? Obviously `X[my_user, my_time, my_feat]` won't work. (Ha! Wouldn't that be nice!) In fact, the easiest answer turns out to be:

```python
Z = X[my_users[:,None], my_times.T[:,:,None], my_feats.transpose(2,0,1)]
```

Urf.

Here's how to do this in Numbat. First, you cast all the arrays to be named tensors, by labeling the axes.

```python
import numbat as nb
u, t, f  = nb.axes()
x        = nb.ntensor(X,        u, t, f)
ny_users = nb.ntensor(my_users, u, f)
ny_times = nb.ntensor(my_times, t, u)
ny_feats = nb.ntensor(my_feats, t, f, u)
```

Then you index in the obvious way: 

```python
z = x(u=ny_users, t=ny_times, f=ny_feats)
```

That's it. That does what you want. Instead of (maddening, slow, tedious, error-prone) manual twiddling to line up the axes, you *label* them and then have the computer line them up for you. Computers are good at that.

### Example 2: Batching

Say that along with `X`, we have some outputs `Y`. For each user and each time, there is some vector of outputs we want to predict. We want to use dead-simple ridge regression, with one regression fit for each user, for each output, and for each of several different regularization constants `R`.

To do this for a single user with a single output and a single regularization constant, remember the [standard formula](https://en.wikipedia.org/wiki/Ridge_regression) that

$w = (X^\top X + rI)^{-1} X^\top y$:

In this simple case, the code is a straightforward translation:

```python
def simple_ridge(X, y, r): 
    n_time, n_feat = x.shape
    n_time2, = y.shape
    assert n_time == n_time2
    
    w = np.linalg.solve(x.T @ x + r * np.eye(n_feat), x.T @ y)
    return w
```

So here's the problem. You've got these three arrays:

```
X.shape == (n_user, n_time, n_feat)
Y.shape == (n_user, n_time, n_pred)
R.shape == (n_reg,)
```

And you'd like to compute some matrix `W` that contains the results of

```python
simple_ridge(X[u,:,:], Y[u,:,p], R[k])
```

for all `u`, `p`, and `k`. How to do that in NumPy?

Well, do you know what `numpy.linalg.solve(a, b)` does when `a` and `b` are high dimensional? The [documentation](https://numpy.org/doc/stable/reference/generated/numpy.linalg.solve.html) is rather hard to parse. The simplest solution turns out to be:

```python
def triple_batched_ridge(X, Y, R):
    n_user,  n_time,  n_feat = X.shape
    n_user2, n_time2, n_pred = Y.shape
    assert n_user == n_user2
    assert n_time == t_time2
    
    XtX = np.sum(X.transpose(0,2,1)[:,:,:,None] * X[:,None,:,:], axis=2) 
    XtY = X.transpose(0,2,1) @ Y
    W = np.linalg.solve(XtX[:,None,:,:] + R[None,:,None,None]*np.eye(n_feat), XtY[:,None,:,:])
    return W
```

Urrrrf.

Even seeing this function, can you tell how the output is laid out? Where in `W` does one find `simple_ridge(X[u,:,:], Y[u,:,p], R[i])`? Would that be in `W[u,p,i]` or `W[i,:,p,u]` or what? The answer turns out to be `W[u,r,:,i]`. Not because you *want* it there, but because of the vagaries of `np.linag.solve` mean that's where it *goes*.

But say you don't want to manually batch things. An alternative is to ask [`jax.vmap`](https://docs.jax.dev/en/latest/_autosummary/jax.vmap.html) to do the batching for you. This is how you'd do that:

```python
triple_batched_ridge_jax = \
    jax.vmap(
        jax.vmap(
            jax.vmap(
                simple_ridge_jax,
                [None, 2, None]), # vmap Y over p
            [0, 0, None]),        # vmap X and Y over u 
        [None, None, 0])          # vmap R over r

W = triple_batched_ridge_jax(X, Y, R)
```

Simple enough, right? 🫡

Maybe. It's also completely wrong. The outermost `vmap` absorbs the first dimension of `Y`, so in the innermost `vmap`, `p` is found in dimension `1`, not dimension `2`. (It's almost like referring to axes by position is confusing!) You also need to mess around with `out_axes` if you want to reproduce the layout of the manually batched function.

So what you actually want is this:

```python
triple_batched_ridge_jax = \
    jax.vmap(
        jax.vmap(
            jax.vmap(
                simple_ridge,
                [None, 1, None], # vmap Y over p
                out_axes=1),     # yeehaw
           [0, 0, None]),        # vmap X and Y over u
       [None, None, 0])          # vmap R over r

W = triple_batched_ridge_jax(X, Y, R)
```

Personally, I think this *is* much better than manual batching. But it still requires you to do a lot of tedious manual tracking of axes as they flow through different operations.

So how would you do this in Numbat? Here's how:

```python
u, t, f, p, i = nb.axes()
x   = nb.ntensor(X, u, t, f)
y   = nb.ntensor(Y, u, t, p)
r   = nb.ntensor(R, i)
fun = nb.lift(simple_ridge, in_axes=[[t,f],[t],[]], out_axes=[f])

w   = fun(x, y, r)
```

Yup, that's it. That works. The `in_axes` argument tells `lift` that  `simple_ridge` should operate on:

1. A 2D array with axes `t` and `f`.
2. A 1D array with axis `t`.
3. A scalar.

And the `out_axes` says that it should return:

1. A 1D array with axis `f`.

When `fun` is finally called, the inputs `x`, `y` and `r` all have named dimensions, so it knows exactly what it needs to do: It should operate on the `t` and `f` axes of `x` and the `t` axis of `y`, and broadcast over everything else. And it should place the output along the `f` axis. 

And where does `simple_ridge(X[u,:,:],Y[u,:,p],R[i])` end up? Well, it's in the only place it could be: `w(u=u, p=p, r=i)`.

(If you find the `lift` syntax clunky, you could write `fun  = nb.lift(simple_ridge, 't f, t, -> f')` instead.)

## Minimal API

If you don't want to learn a lot of features, you can (in principle) do everything with Numbat just using a few functions.

1. Use `ntensor` to create named tensors 
   * To create: `A=ntensor([[1,2,3],[4,5,6]],'i','j')`
   * Use `A.shape` to get the shape (a dict), `A.axes` to get the axes (a set) and `A.ndim` to get the number of dimensions (an int).
   * Use `A(i=i_ind, j=j_ind)` to index.
   * Use `A.numpy('j', 'i')` to convert back to a regular Jax array.
   * Use `A+B`, for (batched) addition, `A*B` for multiplication, etc.

2. Use `dot` to do inner/outer/matrix/tensor products or einstein summation.
   * Use `dot(A,B,C,D)` to sum along all shared axes.
   * * Order doesn't matter!
   * Use `dot(A,B,C,D,keep={'i','j'})` to preserve some shared axes.
   * `A @ B` is equivalent to `dot(A,B)`.

3. Use `batch` to create a batched function
   * Use `batch(fun, {'i', 'j'})(A, B)` to `fun` to the axes `i` and `j` of `A` and `B`, broadcasting over all other axes.

4. Use `vmap` to create a vmapped function.
    * `vmap(fun, {'i', 'j'})(A, B)` applies `fun` to all axes that exist in either `A` or `B` *except* `i` and `j`, broadcasting over `i` and `j`.

5. Use `lift` to wrap Jax functions to operate on `ntensor`s
   * `lift(jnp.matmul, 'i j, j k -> i k')` creates a function that uses `i` and `j` axes of the first argument and the `j` and `k` axes of the second argument.
    
6. Use `grad` and `value_and_grad` to compute gradients.

## Full API docs

API docs are at [https://justindomke.github.io/numbat/](https://justindomke.github.io/numbat/)

## Using Jax features

### Things that work out of the box

`ntensor` is registered with Jax as a Pytree node, so things like `jax.jit` and `jax.tree_flatten` work with `ntensor`s out of the box. For example, this is fine:

```python
import jax
import numbat as nb
x = nb.ntensor([1.,2.,3.],'i')
def fun(x):
   return nb.sum(x)
jax.jit(fun)(x) # works :)
```

### Gradient functions

Gradient functions like `jax.grad` and `jax.value_and_grad` also work out of the box, with one caveat: The *output* of the function to be a jax scalar, and not a `ntensor` scalar. For example, this does not work:

```python
import jax
import numbat as nb
x = nb.ntensor([1.,2.,3.],'i')
def fun(x):
   return nb.sum(x)
jax.grad(fun)(x) # doesn't work :(
```

The problem is that the return value is an `ntensor` with shape `{}`, which `jax.grad` doesn't know what to do with. You can fix this in two ways. First, you can convert a scalar `ntensor` to a Jax scalar using the special `.numpy()` syntax.:

```python
import jax
import numbat as nb
x = nb.ntensor([1.,2.,3.],'i')
def fun(x):
   out = nb.sum(x)
   return out.numpy() # converts to jax scalar 
jax.grad(fun)(x) # works!
```

Alternatively, you can use `numbat.grad` wrapper which does the conversion for you. 

```python
import numbat as nb
x = nb.ntensor([1.,2.,3.],'i')
def fun(x):
   return nb.sum(x) 
nb.grad(fun)(x) # works!
```

### Things that don't work

`jax.vmap` does not work. This is impossible since `jax.vmap` is all based on the positions of axes. Use `numbat.vmap` or `numbat.batch` instead.

## The sharp bits

* If you use the syntax `i,j,k = axes()` to create `Axis` objects, this uses evil magic from the `varname` to try to figure out what the names of `i`, `j`, and `k` are. This package is kinda screwy and might give you errors like `VarnameRetrievingError` or `Couldn't retrieve the call node`. If that happens, try reinstalling varname. Or just give the names explicitly, like `i = Axis('i')`, etc.

* If you're using `jax.tree.*` utilities like `jax.tree.map` these will by default descend into the numpy arrays stored *inside* of `ntensor` objects. You can use `jax.tree.map(..., ..., is_leaf=nb.is_ntensor)` to make sure `ntensor` objects are considered leaves.

## How broadcasting works

You can do broadcasting in three ways:

1. You can use `vmap`:
   - `vmap(f, in_axes)(*args)` maps all arguments in `args` over all axes not in `in_axes`.  
2. You can use `batch`:
   - `batch(f, axes)(*args)` will apply `f` to the axes in `axes`, broadcasting and vmapping over everything else.
3. You can use `wrap`:
   - `wrap(f)(*args, axes=axes)` is equivalent to `batch(f, axes)(*args)`
   - `wrap(f)(*args, vmap=in_axes)` is equivalent to `vmap(f, in_axes)(*args)`
   - If you provide both `axes` *and* `in_axes` then the function checks that all axes are included in one or the other.

## Friends

* [xarray](https://docs.xarray.dev/en/stable/index.html) and the many efforts towards integration with Jax including

    * [xarray_jax](https://github.com/google-deepmind/graphcast/blob/main/graphcast/xarray_jax.py) in [graphcast](https://github.com/google-deepmind/graphcast) 

    * [xarray_jax](https://github.com/allen-adastra/xarray_jax) 

* [named tensors](https://pytorch.org/docs/stable/named_tensor.html) (in PyTorch)

* [Tensor Considered Harmful](http://nlp.seas.harvard.edu/NamedTensor.html)

* [einops](https://github.com/arogozhnikov/einops)

* [xarray-einstats](https://github.com/arviz-devs/xarray-einstats)

* [Nexus](https://github.com/ctongfei/nexus)  

* [Numbat face](https://commons.wikimedia.org/wiki/File:Numbat_Face.jpg)



(Please let me know about any other )