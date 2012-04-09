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
SHOW_TACTICAL_SCORES = True
ts1accu = []
ts2accu = []
tsaccu = []
distrib = []
##### /TODO remove

def army(state, player):
    """ In the given 'state', returns the army (in Unit()) of the 'player' """
    a = []
    for uid, unit in state.tracked_units.iteritems():
        if unit.player == player and unit.name in unit_types.military_set:
            a.append(unit)
    return a

def compute_tactical_distrib(state, player, dm, r, t='Reg'):
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
        distrib.append(tmp[:12])
    ##### /TODO remove
    return 1.0 - s[r]/tot

def extract_tactics_battles(fname, dm, pm=None):
    def detect_attacker(defender, d):
        for k in d:
            if k != defender:
                return k
    def belong(r, defender, attacker, st, dist, t='Reg'):
        """ tells how much a base belongs to the defender """
        # Current version is a "max", because r is can be a CDR or a Reg. 
        # see also data_tools.parse_attacks. TODO
        def positive(x):
            return x >= 0.0
        l_da = filter(positive, [dist.dist(r, rb, t) for rb in st.players_bases[attacker][t]])
        da = 100000000000
        if l_da != []:
            da = min(l_da)
        l_dd = filter(positive, [dist.dist(r, rb, t) for rb in st.players_bases[defender][t]])
        dd = 100000000000
        if l_dd != []:
            dd = min(l_dd)
        if dd <= 0.0 and da <= 0.0:
            return {False: 0.5, True: 0.5}
        elif dd < da:
            return {False: 0.5*dd/da, True: 1.0 - 0.5*dd/da}
        else:
            return {False: 1.0 - 0.5*da/dd, True: 0.5*da/dd}
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
            b1 = belong(cdr, defender, attacker, st, dm, t='CDR')
            b2 = belong(reg, defender, attacker, st, dm, t='Reg')
            # pick the max score of "this region belongs to the defender"
            if b1[True] > b2[True]:
                tmp[2]['belong'] = b1
            else:
                tmp[2]['belong'] = b2
            # use an alternative tactical score: mean ground (pathfinding)
            # distance of the region to the defender's army
            ts1 = compute_tactical_distrib(st, defender, dm, cdr, t='CDR')
            ts2 = compute_tactical_distrib(st, defender, dm, reg, t='Reg')
            tmp[2]['tactic'] = max(ts1, ts2)
            ##### TODO remove
            if SHOW_TACTICAL_SCORES:
                ts1accu.append(ts1)
                ts2accu.append(ts2)
                tsaccu.append(tmp[2]['tactic'])
            ##### /TODO remove
            
            battles.append((tmp[0], tmp[2]))
    return battles

class TacticalModel:
    """
    For all region r we have:
        A (Attack) in true/false
        EI (Economical importance) in [0..1] for the player considered
        TI (Tactical importance) in [0..1] for the player considered
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
        self.Atrue_knowing_EI_TI_B = np.ndarray(shape=(10,10,3), dtype='float')
        self.Atrue_knowing_EI_TI_B.fill(ADD_SMOOTH)
        # H_knowing_AD_GD_ID[ground/air/drop/invis][ad][gd][id] = proba
        self.H_knowing_AD_GD_ID = np.ndarray(shape=(4,3,3,3), dtype='float')
        self.H_knowing_AD_GD_ID.fill(ADD_SMOOTH)
    def __repr__(self):
        print "*** P(A=true | EI, TI, B) ***"
        print self.Atrue_knowing_EI_TI_B
        print "*** P(H | AD, GD, ID) ***"
        print self.H_knowing_AD_GD_ID
    def train(self, battles):
        """
        (['GroundAttack'], {'detect': 0.0, 'eco': 0.0, 'belong': {False: 1.0, True: 0.0}, 'air': 0.0, 'tactic': 238768128.0, 'ground': 200.0})
        (['GroundAttack'], {'detect': 0.0, 'eco': 0.0, 'belong': {False: 0.0949367088607595, True: 0.9050632911392404}, 'air': 583.3333, 'tactic': 348216770.56, 'ground': 1183.3333})
        ([], {'detect': 0.0, 'eco': 0.0, 'belong': {False: 0.0949367088607595, True: 0.9050632911392404}, 'air': 583.3333, 'tactic': 422453411.84, 'ground': 583.3333})
        (['GroundAttack'], {'detect': 0.0, 'eco': 0.0, 'belong': {False: 0.0949367088607595, True: 0.9050632911392404}, 'air': 0.0, 'tactic': 348216770.56, 'ground': 1200.0})
        (['GroundAttack'], {'detect': 0.0, 'eco': 0.0, 'belong': {False: 0.0949367088607595, True: 0.9050632911392404}, 'air': 0.0, 'tactic': 349539205.12, 'ground': 1500.0})
        (['GroundAttack'], {'detect': 0.0, 'eco': 0.3, 'belong': {False: 0.0, True: 1.0}, 'air': 2916.6667, 'tactic': 505865584.64, 'ground': 4131.6667})
        (['GroundAttack'], {'detect': 0.0, 'eco': 0.0, 'belong': {False: 0.9160789844851904, True: 0.08392101551480959}, 'air': 0.0, 'tactic': 172366888.96, 'ground': 200.0})
        (['GroundAttack'], {'detect': 0.0, 'eco': 0.0, 'belong': {False: 0.2724373576309795, True: 0.7275626423690205}, 'air': 1458.3333, 'tactic': 217653329.92, 'ground': 1858.3333})
        (['GroundAttack'], {'detect': 1.0, 'eco': 0.0, 'belong': {False: 0.0, True: 1.0}, 'air': 2041.6667, 'tactic': 315149639.68, 'ground': 2141.6667})
        (['GroundAttack'], {'detect': 0.0, 'eco': 0.0, 'belong': {False: 0.46119324181626187, True: 0.5388067581837381}, 'air': 3500.0, 'tactic': 260198359.04, 'ground': 3600.0})
        """
        #for b in battles:
        #    for attack_type in b[0]:
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
    if SHOW_TACTICAL_SCORES:
        import matplotlib.pyplot as plt
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
    ##### /TODO remove


