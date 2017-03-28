
import numpy as np

# TODO consider moving these to another module
np3481 = {'DM6', 'DL5', 'VM2', 'VM7'}
of_interest = set(np3481)
of_interest.remove('VM2')

def conc_range(odor, low, high):
    """
    Returns a tuple of pairs like ((odor, high), (odor, high + 1), ... (odor, low))
    """

    assert low <= 0 and high <= 0,  'low and high should be powers of ten that reflect v/v dilution'
    assert low <= high, 'low should be <= high'

    return tuple(zip((odor,) * (high - low + 1), range(low, high + 1)))

def add_glom(range_tuples, glom):
    """
    Input: ((odor, power of 10), ...)
    Output: ( ((odor, power of 10), glom), ...)
    """

    return tuple(map(lambda t: (t, glom), range_tuples))

# TODO calculate this kind of stuff from HC

# TODO also worth including odors that activate or inhibit uniquely, to the extent that no
# other glomeruli activated or inhibited are likely to be in the same plane
# more complicated though

# consider defining these over ranges rather than integer powers of 10, though for now,
# the latter is sufficient for all of our experimental practices

add = lambda o, l, h, g: add_glom(conc_range(o, l, h), g)

'''
# should add this information separately later, but for now i just want to identify glomeruli
# with good performance
ua_tuples = [add('pentanoic acid', -4, -2, 'DM6'), # conc?
             add('trans-2-hexenal', -8, -6, 'DL5'),
             add('2-butanone', -8, -4, 'VM7')]
'''
ua_tuples = [add('pentanoic acid', -3, -3, 'DM6'), # conc?
             add('trans-2-hexenal', -6, -6, 'DL5'),
             add('2-butanone', -6, -6, 'VM7')]

# VM2 doesn't really have a private (ZG)? iba but not 2-but
# entries should satisfy: <odor (key)> uniquely activates <glomerulus (value)>
uniquely_activates = dict()

for group in ua_tuples:
    for t in group:
        uniquely_activates[t[0]] = t[1]

# TODO change to reflect that it likely is not *uniquely* inhibiting
# TODO only use highest concentration private odor for identifying glomeruli 
# TODO concs for these? unique?
ui_tuples = [add('isobutyl acetate', -3, -3, 'DM6'),
             add('2-heptanone', -2, -2, 'DL5'),
             add('pentanoic acid', -5, -2, 'VM7')]

# entries should satisfy: <odor (key)> uniquely inhibits <glomerulus (value)>
uniquely_inhibits = dict()

for group in ui_tuples:
    for t in group:
        uniquely_inhibits[t[0]] = t[1]

# TODO print odor panel to check for float parsing issues

# TODO might want to store actual concentrations and not just the exponents
def str2pair(odor_string):
    """ 
    Takes a string with an odor name followed by a concentration in scientific notation
    and returns a tuple of (odor name, power of ten reflecting concentration)

    -works best under assumption all odors in concentrations of powers of 10 (v/v dilution)
    -notation for concentration needs to be last element of string, with no whitespace
     in the number, and whitespace immediately to the left of it
    """

    split = odor_string.split()

    return (' '.join(split[:-1]), int(np.round(np.log10(float(split[-1])))))


def pair2str(odor_pair):
    """
    Inverse of str2pair.
    """
    odor, c = odor_pair
    return odor + ' 1e' + str(c)


def is_private(odor):

    if type(odor) is str:
        odor = str2pair(odor)

    return odor in uniquely_activates

# accurate to say uniquely?
def is_uniquely_inhibiting(odor):

    if type(odor) is str:
        odor = str2pair(odor)

    return odor in uniquely_inhibits

def in_np3481(glom):
    return glom.upper() in np3481

def is_of_interest(glom):
    return glom.upper() in of_interest

