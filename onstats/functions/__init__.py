"""Stats functions subpackage.

Define here stats functions to be plugged into the Stats class at creation time.
The functions can be defined in any module within this subpackage.

Stats functions must have this signature:

    func_name(self, dset, **kwargs)

with `self` always defined as the first argument. The second argument must be the
dataset object `dset` to be reduced, followed by additional keyword arguments to be
passed to the stats functions.

"""
