
import shelve
from functools import wraps
import inspect
from os.path import exists, isfile

# this SO post explains why functools wraps is a good thing
# https://stackoverflow.com/questions/308999/what-does-functools-wraps-do

required_args = ('key', 'clear', 'cache_path')

# TODO ?
#def clear():
#    if 


def full_key(func, key):
    #filename = os.path.split(inspect.stack()[-1].filename)[-1]
    #os.path.abspath(rel_parent_file)
    caller_frame = inspect.stack()[-1]
    filename = inspect.getmodule(caller_frame)
    print(filename)
    print(func.__name__)
    return '#'.join([filename, func.__name__, key])


# TODO test
def check_argname_conflicts(func):
    # will be removed at some point in Python 3.6 (but seems fine for now)
    args = inspect.getargspec(func).args
    for a in required_args:
        if a in args:
            raise ValueError('functions with @cache decorator can not have ' + a + ' as ' + \
                    'an argument name. it is used by the wrapper.')


# TODO how does shelve behave if file is there? is not writable? is directory?
# TODO just delete if clear check and get rid of clear here?
def cache(func):
    check_argname_conflicts(func)

    @wraps(func)
    def with_caching(*args, **kwargs):
        for a in required_args:
            if a not in kwargs:
                raise ValueError('need to manually set key (something hashable), clear ' + \
                        '(True/False), and path for cache file. OK to use same key across' + \
                        'functions.')

        # TODO check writable?
        with shelve.open(cache_name) as store:
            results = func(*args, **kwargs)
            # TODO anything important not serializable?
            store[kwargs[full_key(func, key)]] = results
            return results
    
    return with_caching
