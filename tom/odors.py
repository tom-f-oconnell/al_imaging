
# TODO do concentration parsing w/ regex later

# make these frozensets
np3481 = {'DM6', 'DL5', 'VM2', 'VM7'}
of_interest = set(np3481).remove('VM2')

# TODO calculate this kind of stuff from HC

# uniquely?
# VM2 doesn't really have a private (ZG)? iba but not 2-but
private = {'pentanoic acid': 'DM6',  # ?
           'trans-2-hexenal': 'DL5', # ~6-8
           '2-butanone': 'VM7'} # < -4
           
inhibiting = {'isobutyl acetate': 'DM6',
              '2-heptanone':   'DL5',
              'pentanoic acid': 'VM7'}

def in_np3481(glom):
    return glom.upper() in np3481

def of_interest(glom):
    return glom.upper() in of_interest

