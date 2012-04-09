import sys, re
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
                if type(v) == str:
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

def parse_attacks(s):
    """ return a tuple (types_list, init_position_tuple, scores_dict) """
    l = s.split(',')
    l[3] = l[3][1:]
    t = []
    if len(l[3]) > 0:
        for i in range(3,len(l)):
            if l[i][-1] != ')':
                t.append(l[i])
            else:
                t.append(l[i][:-1])
                break
    tmpp = re.search('\((\d+),(\d+)\)', s)
    p = (int(tmpp.group(1)), int(tmpp.group(2)))
    tmpscores = re.search('\((\d+\.\d\d\d\d|-1.#IND),(\d+\.\d\d\d\d|-1.#IND),(\d+\.\d\d\d\d|-1.#IND),(\d+\.\d\d\d\d|-1.#IND),(\d+\.\d\d\d\d|-1.#IND),(\d+\.\d\d\d\d|-1.#IND),(\d+\.\d\d\d\d|-1.#IND),(\d+\.\d\d\d\d|-1.#IND),(\d+\.\d\d\d\d|-1.#IND),(\d+\.\d\d\d\d|-1.#IND)\)', s)
    sc = []
    for i in range(1,11):
        if tmpscores.group(i) == '-1.#IND':
            sc.append(0.0) # TODO verif
        else:
            sc.append(float(tmpscores.group(i)))
    # all these "max" in the score should instead be 2 scores leading to 
    # two different sets of learned parameters TODO
    scores = {'ground': max(sc[0], sc[1]),
            # ground score is unit_types.score_unit applied to units who/which
            # can fire on ground units in the region (for the defender)
              'air': max(sc[2], sc[3]),
            # air score is unit_types.score_unit applied to units who/which
            # can fire on air units in the region (for the defender)
              'detect': max(sc[4], sc[5]),
            # detect score is the number of units with detection in the region
              'eco': max(sc[6], sc[7]),
            # eco score is the number of working peons in the region 
            # divided per the total number of working peons for the defender
              'tactic': max(sc[8], sc[9])}
            # tactic score is proportional to the sum of the square distances
            # to bases + to the region closest to the mean position of the army
            # i.e. tactic score for a region r is prop. to: 
            #   \sum_{b \in bases}[dist(r, b)^2]
            # + dist(r, 1/#units * \sum_{pu \in pos_units}pu)^2
    # (types_list, init_position_tuple, scores_dict)
    return (t, p, scores)

def players_races(f):
    """ returns a dict filled with {player_id : race} """
    r = {}
    read = False
    for line in f:
        if "The following players are in this replay:" in line:
            read = True
        elif "Begin replay data:" in line:
            break
        elif read:
            l = line.split(',')
            r[l[0]] = l[2][1]
    return r
