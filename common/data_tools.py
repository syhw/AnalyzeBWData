import sys
try:
    import numpy as np
except:
    print "you need numpy"
    sys.exit(-1)

def features_scaling(tt):
    """ find max and compute mean for each rows of tt (numpy array) """
    mx = tt.max(axis=0)
    # some features'values are always 0
    for i, e in enumerate(mx):
        if e == 0.0:
            mx[i] = 1.0
    me = tt.mean(axis=0)
    # divide each feature by its max
    return (tt - me) / mx

def consume_dict(delim, s, f=lambda x: x):
    """ returns (delim[1]_pos, dict) with dict starting at s[0] 
    and using the optional f argument function on values while building dict"""
    d = {}
    k = ''
    v = ''
    i = 0
    key = True
    value = False
    while len(s) > i:
        char = s[i]
        if char == delim[0]:
            cpos, v = consume_dict(delim, s[i+1:], f)
            i += cpos+1
            d[k] = v
            k = ''
            v = ''
        elif char == delim[1]:
            if k != '':
                if type(v) == type(''):
                    d[k] = f(v)
                else:
                    d[k] = v
            return i, d
        elif char == ':':
            v = ''
            value = True
            key = False
        elif char == ',':
            if v != '':
                d[k] = f(v)
                k = ''
                v = ''
            key = True
            value = False
        else:
            if key:
                k += char
            elif value:
                v += char
        i += 1
    if k != '':
        if type(v) == type(''):
            d[k] = f(v)
        else:
            d[k] = v
    return i, d

def parse_dicts(s, f=lambda x: x):
    """ returns a list of all python-like "{'e':1}" dictionaries of the s """
    r = []
    rs = s+' '
    opening = rs.find('{')
    while opening > -1:
        rs = rs[opening+1:]
        closing, d = consume_dict('{}', rs, f)
        r.append(d)
        rs = rs[closing+1:]
        opening = rs.find('{')
    return r

