import sys, os, pickle, copy, itertools, functools
from common import data_tools
from common import unit_types
from common import attack_tools
from common import state_tools
from common.position_tools import PositionMapper
from common.position_tools import DistancesMaps
try:
    import numpy as np
except:
    print "you need numpy"
    sys.exit(-1)

ADD_SMOOTH = 1.0 # Laplace smoothing, could be less
TACT_PARAM = 1.6 # power of the distance of units to/from regions
# 1.6 means than a region which is at distance 1 of the two halves of the army
# of the player is 1.5 more important than one at distance 2 of the full army
##### TODO remove
SHOW_TACTICAL_SCORES = False
SHOW_ECO_SCORES = False
ts1accu = []
ts2accu = []
tsaccu = []
distrib = []
esaccu = []
##### /TODO remove

def army(state, player):
    """ In the given 'state', returns the army (in Unit()) of the 'player' """
    a = []
    for uid, unit in state.tracked_units.iteritems():
        if unit.player == player and unit.name in unit_types.military_set:
            a.append(unit)
    return a

def compute_tactical_score(state, player, dm, r, t='Reg'):
    """ 
    Computes the tactical score of Reg/CDR 'r', according to 'state'
    and for 'player' ('dm' is the distance map and 't' the type of 'r'):
    1.0 - tactical_score[reg] = 
        \sum_{unit}[score[unit]*dist(unit,reg)^TACT_PARAM] (normalized)
    Higher = better (higher <=> closer to the mean square of the army)
    """
    # TODO review this heuristic
    tot = 0.00000000001
    a = army(state, player)
    s = {}
    ref = dm.dist_Reg
    if t == 'CDR':
        ref = dm.dist_CDR
    for tmpr in ref:
        s[tmpr] = 0.0
        for unit in a:
            d = dm.dist(unit[t], tmpr, t)
            if d > 0.0:
                s[tmpr] += unit_types.score_unit(unit.name)*(d**TACT_PARAM)
            else:
                s[tmpr] += unit_types.score_unit(unit.name)*(dm.max_dist**TACT_PARAM)
        tot += s[tmpr]
    ##### TODO remove
    if SHOW_TACTICAL_SCORES:
        tmp = []
        for k,v in s.iteritems():
            tmp.append(1.0 - v)
        tmp.sort()
        distrib.append(tmp[-12:])
    ##### /TODO remove
    return 1.0 - s[r]/tot

def belong_distrib(r, defender, attacker, st, dm, t='Reg'):
    """ tells how much a base belongs to the defender """
    # Current version is a "max", because r can be a CDR or a Reg. 
    # see also data_tools.parse_attacks.
    def positive(x):
        return x >= 0.0
    l_da = filter(positive, [dm.dist(r, rb, t) for rb in st.players_bases[attacker][t]])
    da = 100000000000
    if l_da != []:
        da = min(l_da)
    l_dd = filter(positive, [dm.dist(r, rb, t) for rb in st.players_bases[defender][t]])
    dd = 100000000000
    if l_dd != []:
        dd = min(l_dd)
    # indice values: 0 for False and 1 for True
    if dd <= 0.0 and da <= 0.0: # really contested region
        return {0: 0.5, 1: 0.5}
    elif dd < da: # distance to defenser's base is closer (can be a def's base)
        return {0: 0.5*dd/da, 1: 1.0 - 0.5*dd/da}
    else: # distance to attacker's base is closer (can be an attacker's base)
        return {0: 1.0 - 0.5*da/dd, 1: 0.5*da/dd}

def where_bins(s, bins):
    ind = 0
    for i,v in enumerate(bins):
        if s > v:
            ind = i
        else:
            return ind
    return ind # defensive prog

def tactic_distrib(score):
    # TODO revise distrib into true distrib?
    bins = [0.0, 0.9651523609362167, 0.9760342259222318, 0.9824477540926082, 0.9879026329442978, 1.0] # equitable repartition of # of attacks in 5 bins
    d = {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0}
    d[where_bins(score, bins)] = 1.0
    return d

def eco_distrib(score):
    # TODO revise distrib into true distrib?
    bins = [0.0, 0.05, 0.66, 1.0] # no eco, small eco, more than 2/3rd of total
    d = {0: 0.0, 1: 0.0, 2: 0.0}
    d[where_bins(score, bins)] = 1.0
    return d

def score_units_criterium(units, s, f):
    sc = 0.0
    for u in units:
        if f(u, s):
            sc += unit_types.score_unit(u)
    return sc

def score_air(units):
    return score_units_criterium(units, unit_types.flying_set,
            lambda x,y: x in y)

def score_ground(units):
    return score_units_criterium(units, unit_types.flying_set,
            lambda x,y: x not in y)

def detect_distrib(score):
    # TODO revise distrib into true distrib?
    bins = [0.0, 0.99, 1.99, 100000.0] # none, one, many
    d = {0: 0.0, 1: 0.0, 2: 0.0}
    d[where_bins(score, bins)] = 1.0
    return d

def units_distrib(score): # score is given relative to attackers force
    # TODO revise distrib into true distrib?
    bins = [0.0, 0.05, 0.5, 1.0e80] # no defense (20x smaller than attacker)
    #small defense (up to 2x smaller than attacker's force), >2x => big defense
    d = {0: 0.0, 1: 0.0, 2: 0.0}
    d[where_bins(score, bins)] = 1.0
    return d

def extract_tactics_battles(fname, dm, pm=None):
    def detect_attacker(defender, d):
        for k in d:
            if k != defender:
                return k
    obs = attack_tools.Observers()
    st = state_tools.GameState()
    st.track_loc(open(fname[:-3]+'rld'))
    battles = []
    f = open(fname)
    for line in f:
        line = line.rstrip('\r\n')
        obs.detect_observers(line)
        st.update(line)
        if 'IsAttacked' in line:
            tmp = data_tools.parse_attacks(line)
            cdr = pm.get_CDR(tmp[1][0], tmp[1][1])
            reg = pm.get_Reg(tmp[1][0], tmp[1][1])
            units = data_tools.parse_dicts(line)
            units = obs.heuristics_remove_observers(units)
            defender = line.split(',')[1]
            attacker = detect_attacker(defender, units[0])
            b1 = belong_distrib(cdr, defender, attacker, st, dm, t='CDR')
            b2 = belong_distrib(reg, defender, attacker, st, dm, t='Reg')
            # pick the max score of "this region belongs to the defender"
            if b1[1] > b2[1]:
                tmp[2]['belong'] = b1
            else:
                tmp[2]['belong'] = b2
            # use an alternative tactical score: mean ground (pathfinding)
            # distance of the region to the defender's army
            ts1 = compute_tactical_score(st, defender, dm, cdr, t='CDR')
            ts2 = compute_tactical_score(st, defender, dm, reg, t='Reg')
            tmp[2]['tactic'] = max(ts1, ts2)
            ##### TODO remove
            if SHOW_TACTICAL_SCORES:
                ts1accu.append(ts1)
                ts2accu.append(ts2)
                tsaccu.append(tmp[2]['tactic'])
            if SHOW_ECO_SCORES:
                esaccu.append(tmp[2]['eco'])
            ##### /TODO remove
            tmp[2]['tactic'] = tactic_distrib(tmp[2]['tactic'])
            tmp[2]['eco'] = eco_distrib(tmp[2]['eco'])
            tmp[2]['detect'] = detect_distrib(tmp[2]['detect'])
            tmp[2]['air'] = units_distrib(tmp[2]['air'] / (0.1+score_air(units[0][attacker])))
            tmp[2]['ground'] = units_distrib(tmp[2]['ground'] / (0.1+score_ground(units[0][attacker])))
            battles.append((tmp[0], tmp[2]))
    return battles

class TacticalModel:
    """
    For all region r we have:
        A (Attack) in true/false
        EI (Economical importance) in {0, 1, 2} for the player considered
        TI (Tactical importance) in {0, 1, 2, 3, 4} for the player considered
        B (Belongs) in {True/False} for the player considered
            # P(A, EI, TI, B) = P(EI|A)P(TI|A)P(B|A)P(A)
        ==> P(A, EI, TI, B) = P(EI)P(TI)P(B)P(A | EI, TI, B)
        ?: P([A=1] | EEI, TI) = sum_B[P([A=1] | EI, TI, B).P(B)]
        P(B=True) = 1.0 iff r si one of the bases of the player considered
        P(B=False) = 1.0 iff r si one of the bases of the ennemy of the player
        P(B=True) prop to min_{base \in players'bases}(dist(r, base))
        ex.: P(B=True) = 0.5 in the middle of the map

        H (How) in {Ground, Air, Drop, Invisible}
        AD (Air defense) in {0, 1, 2} (0: no defense, 1: light defense compared
        GD (Ground defense) in {0, 1, 2}  ..to the attacker, 2: heavy defense)
        ID (Invisible defense = #detectors) in {0, 1, 2+}
            # P(H, AD, GD, ID) = P(AD|H)P(GD|H)P(ID|H)P(H)
        ==> P(H, AD, GD, ID) = P(AD)P(GD)P(ID)P(H | AD, GD, ID)
        ?: P(H) = sum_{AD}[P(AD) sum_{GD}[P(GD) sum_{ID}[P(ID)P(H | AD, GD, ID)]]]
        P(A/G/ID=0) = 1.0 iff r has no defense against this type of attack
        P(A/G/ID=1) = 1.0 iff r has less than one half the score of the assaillant
                              on a given attack type
        P(A/G/ID=2) = 1.0 iff r has more than one half the score of the assaillant
                              on a given attack type
    """

    def __init__(self):
        # Atrue_knowing_EI_TI_B[ei][ti][b] = proba
        self.Atrue_knowing_EI_TI_B = np.ndarray(shape=(3,5,2), dtype='float')
        self.Atrue_knowing_EI_TI_B.fill(ADD_SMOOTH)
        # H_knowing_AD_GD_ID[ground/air/drop/invis][ad][gd][id] = proba
        self.H_knowing_AD_GD_ID = np.ndarray(shape=(4,3,3,3), dtype='float')
        self.H_knowing_AD_GD_ID.fill(ADD_SMOOTH)

    def __repr__(self):
        s = "*** P(A=true | EI, TI, B) ***\n"
        s += self.Atrue_knowing_EI_TI_B.__repr__() + '\n'
        s += "*** P(H | AD, GD, ID) ***\n"
        s += self.H_knowing_AD_GD_ID.__repr__()
        return s

    def train(self, battles):
        """
        fills Atrue_knowing_EI_TI_B and H_knowing_AD_GD_ID according to battles
        """
        def attack_type_to_ind(at):
            if at == 'GroundAttack':
                return 0
            elif at == 'AirAttack':
                return 1
            elif at == 'DropAttack':
                return 2
            elif at == 'InvisAttack':
                return 3
            else:
                print "Not a good attack type label"
                raise TypeError

        for b in battles:
            #print b
            for keco,veco in b[1]['eco'].iteritems():
                for ktac,vtac in b[1]['tactic'].iteritems():
                    for kbel,vbel in b[1]['belong'].iteritems():
                        self.Atrue_knowing_EI_TI_B[keco, ktac, kbel] += veco*vtac*vbel
            for attack_type in b[0]:
                for kair,vair in b[1]['air'].iteritems():
                    for kground,vground in b[1]['ground'].iteritems():
                        for kdetect,vdetect in b[1]['detect'].iteritems():
                            self.H_knowing_AD_GD_ID[attack_type_to_ind(attack_type), kair, kground, kdetect] += vair*vground*vdetect
        print "I've seen", len(battles), "battles"


if __name__ == "__main__":
    # serialize?
    fnamelist = []
    if sys.argv[1] == '-d':
        import glob
        fnamelist = glob.iglob(sys.argv[2] + '/*.rgd')
    else:
        fnamelist = sys.argv[1:]
    battles = []
    for fname in fnamelist:
        f = open(fname)
        floc = open(fname[:-3]+'rld')
        dm = DistancesMaps(floc)
        floc.close()
        floc = open(fname[:-3]+'rld')
        pm = PositionMapper(floc)
        players_races = data_tools.players_races(f)
        battles.extend(extract_tactics_battles(fname, dm, pm))
    tactics = TacticalModel()
    tactics.train(battles)
    print tactics
    ##### TODO remove
    import matplotlib.pyplot as plt
    if SHOW_TACTICAL_SCORES:
        plt.hist(tsaccu,10)
        plt.show()
        plt.hist(ts1accu,10)
        plt.show()
        plt.hist(ts2accu,10)
        plt.show()
        s = [0.0 for i in range(len(distrib[0]))]
        tot = 0.0
        for d in distrib:
            for i,e in enumerate(d):
                s[i] += e
                tot += e
        m = [s[i]/tot for i in range(len(distrib[0]))]
        x = [i for i in range(len(distrib[0]))]
        plt.bar(x, m)
        plt.show()
        n, bins, patches = plt.hist(tsaccu, 5)#, log=True)
        plt.show()
        print bins
        tsaccu.sort()
        bins = [0.0, tsaccu[len(tsaccu)/5], tsaccu[2*len(tsaccu)/5], tsaccu[3*len(tsaccu)/5], tsaccu[4*len(tsaccu)/5], 1.0]
        plt.hist(tsaccu, bins)
        plt.show()
        print bins
    if SHOW_ECO_SCORES:
        n, bins, patches = plt.hist(esaccu, 3, log=True)
        plt.show()
        print bins
        esaccu.sort()
        while esaccu.count(0.0):
            esaccu.remove(0.0)
        first_val = filter(lambda x: x>0.05, esaccu)[0]
        bins = [first_val, esaccu[len(esaccu)/2], 1.0]
        plt.hist(esaccu, bins)
        plt.show()
        print [0.0] + bins
    ##### /TODO remove


