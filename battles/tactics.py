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

def extract_tactics_battles(fname, dm, pm=None):
    def detect_attacker(defender, d):
        for k in d:
            if k != defender:
                return k
    def belong(r, defender, attacker, state, dist, t='Reg'):
        """ tells how much a base belongs to the defender """
        # Current version is a "max", because r is can be a CDR or a Reg. 
        # see also data_tools.parse_attacks. TODO
        da = min([dist.dist(r, rb, t) for rb in state.players_bases[attacker][t]])
        dd = min([dist.dist(r, rb, t) for rb in state.players_bases[defender][t]])
        if dd < da:
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
            print tmp[1]
            cdr = pm.get_CDR(tmp[1][0], tmp[1][1])
            reg = pm.get_Reg(tmp[1][0], tmp[1][1])
            units = data_tools.parse_dicts(line)
            units = obs.heuristics_remove_observers(units)
            defender = line.split(',')[1]
            attacker = detect_attacker(defender, units[0])
            print cdr
            print reg
            print st.players_bases
            b1 = belong(cdr, defender, attacker, st, dm, t='CDR')
            b2 = belong(reg, defender, attacker, st, dm, t='Reg')
            if b1[True] > b2[True]:
                tmp[2]['belong'] = b1
            else:
                tmp[2]['belong'] = b2
            battles.append((tmp[0], tmp[2]))
    return battles

class TacticalModel:
    """
    For all region r we have:
        A (Attack) = true/false
        EI (Economical importance) = [[0..9]] for the player considered
        TI (Tactical importance) = [[0..9]] for the player considered
        B (Belongs) = {True/False} for the player considered
            # P(A, EI, TI, B) = P(EI|A)P(TI|A)P(B|A)P(A)
        ==> P(A, EI, TI, B) = P(EI)P(TI)P(B)P(A | EI, TI, B)
        ?: P([A=1] | EEI, TI) = sum_B[P([A=1] | EI, TI, B).P(B)]
        P(B=True) = 1.0 iff r si one of the bases of the player considered
        P(B=False) = 1.0 iff r si one of the bases of the ennemy of the player
        P(B=True) prop to min_{base \in players'bases}(dist(r, base))
        ex.: P(B=True) = 0.5 in the middle of the map

        H (How) = {Ground, Air, Drop, Invisible}
        AD (Air defense) = {0, 1, 2}
        GD (Ground defense) = {0, 1, 2}
        ID (Invisible defense = detectors) = {0, 1, 2}
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
    def train(self, battles):
        for b in battles:
            print b

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


