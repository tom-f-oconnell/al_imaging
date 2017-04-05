
from collections import Hashable

# TODO are there any things that are not hashable that meet the percache condition on repr?
# TODO any things that are hashable but fail that condition?
# ...seems hard / impossible to directly check percache condition
def good_repr_for_hashing(x):
    try:
        # TODO need to handle empty iterables separately? at least their repr should be 
        # unambiguous...
        # the statement that should fail if x isn't an iterable
        elements_hashable = [good_repr_for_hashing(e) for e in x]

        return all(elements_hashable)
        
    # was not an iterable
    # TODO is there anything else that could have caused this error? safe?
    except TypeError:
        return isinstance(x, Hashable)

def check_args_hashable(func):
    def f(*args, **kwargs):
        # TODO how are args formatted exactly?
        # TODO does percache also use kwargs? if so, check those as well

        # TODO remove try block. for debugging.
        try:
            v = good_repr_for_hashing(args)
        except RecursionError:
            print(args)
            raise

        assert v, 'passed instance of ' + str(type(args)) + \
                ' seems to have unsuitable repr for use with percache. see docs.'
        return f(*args, **kwargs)
    return f

# TODO test both
