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

testing = True

ADD_SMOOTH = 0.5 # Laplace smoothing, could be less
TACT_PARAM = 1.6 # power of the distance of units to/from regions
NUMBER_OF_TEST_GAMES = 10 # number of games to evaluates the tactical model on
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

def select(state, player, inset):
    """ In the given 'state', returns the units in 'inset' of the 'player' """
    x = []
    for uid, unit in state.tracked_units.iteritems():
        if unit.player == player and unit.name in inset:
            x.append(unit)
    return x

def army(state, player):
    """ In the given 'state', returns the army (in Unit()) of the 'player' """
    return select(state, player, unit_types.military_set)

def compute_scores(state, player, dm, inset, scoring_f, t='Reg'):
    """ computes the economical scores of all regions of type t and returns 
    (scores_dict, total) """
    s = {}
    tot = 0.00000000001
    x = select(state, player, inset)
    for tmpr in dm.list_regions(t): # yes, it's dumb, but I don't want a sparse dict()
        s[tmpr] = 0.0               # even though I could get(x, default)...
        for unit in x:
            if unit[t] == tmpr:
                s[tmpr] += scoring_f(unit)
        tot += s[tmpr]
    return (s, tot)

def compute_tactical_scores(state, player, dm, t='Reg'):
    """ computes the tactical scores of all regions of type t and returns 
    (scores_dict, total) """
    s = {}
    tot = 0.00000000001
    a = army(state, player)
    for tmpr in dm.list_regions(t):
        s[tmpr] = 0.0
        for unit in a:
            if unit[t] == -1:
                continue
            d = dm.dist(unit[t], tmpr, t)
            if d > 0.0:
                s[tmpr] += unit_types.score_unit(unit.name)*(d**TACT_PARAM)
            else:
                s[tmpr] += unit_types.score_unit(unit.name)*(dm.max_dist**TACT_PARAM)
        tot += s[tmpr]
    return (s, tot)

def compute_tactical_score(state, player, dm, r, t='Reg'):
    """ 
    Computes the tactical score of Reg/CDR 'r', according to 'state'
    and for 'player' ('dm' is the distance map and 't' the type of 'r'):
    1.0 - tactical_score[reg] = 
        \sum_{unit}[score[unit]*dist(unit,reg)^TACT_PARAM] (normalized)
    Higher = better (higher <=> closer to the mean square of the army)
    """
    # TODO review this heuristic
    s, tot = compute_tactical_scores(state, player, dm, t)
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
    def make_positive(x):
        if x < 0.0: # it's an island
            return 11500.0 # that's 256(max map size)*32(pix/tile)*sqrt(2)
        return x
    l_da = map(make_positive, [dm.dist(r, rb, t) for rb in st.players_bases[attacker][t]])
    da = 100000000000
    if l_da != []:
        da = min(l_da)
    l_dd = map(make_positive, [dm.dist(r, rb, t) for rb in st.players_bases[defender][t]])
    dd = 100000000000
    if l_dd != []:
        dd = min(l_dd)
    # indice values: 0 for False and 1 for True
    if dd <= 0.0 and da <= 0.0: # really contested region, one base for each or island w/o base!
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

def detect_attacker(defender, d):
    for k in d:
        if k != defender:
            return k

def extract_tactics_battles(fname, dm, pm=None):
    """ 
    Extract all attacks and tactics from the file named fname,
    with the distance map (between regions) dm, and the positional mapper pm
    returns a list of battles [([types], scoredict, cdr, reg)]
    """
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
            battles.append((tmp[0], tmp[2], {'Reg': reg, "CDR": cdr}))
            #              (list of types, dict of distribs, CDR, Reg)
    return battles

def extract_tests(fname, dm, pm=None):
    """
    Extract tests situations from the file named fname,
    with the distance map (between regions) dm, and the positional mapper pm
    returns a list of dicts of regions scores:
    [{'Reg': {score_type: {reg: distirb}},'CDR': {score_type: {reg: distirb}}}]
    """
    obs = attack_tools.Observers()
    st = state_tools.GameState()
    st.track_loc(open(fname[:-3]+'rld'))
    tests = []
    f = open(fname)
    for line in f:
        line = line.rstrip('\r\n')
        obs.detect_observers(line)
        st.update(line)
        if 'IsAttacked' in line:
            units = data_tools.parse_dicts(line)
            units = obs.heuristics_remove_observers(units)
            defender = line.split(',')[1]
            attacker = detect_attacker(defender, units[0])
            tmp = {'Reg': {}, 'CDR': {}}
            for rt in tmp:
                s, tot = compute_scores(st, defender, dm, unit_types.workers, lambda x: 1.0, t=rt)
                for k in s:
                    s[k] = eco_distrib(s[k]/tot)
                tmp[rt]['eco'] = s
                s, tot = compute_tactical_scores(st, defender, dm, t=rt)
                for k in s:
                    s[k] = tactic_distrib(s[k]/tot)
                tmp[rt]['tactic'] = s
                tmp[rt]['belong'] = {}
                for r in dm.list_regions(rt):
                    tmp[rt]['belong'][r] = belong_distrib(r, defender, attacker, st, dm, rt)
                s, tot = compute_scores(st, defender, dm, unit_types.detectors_set, lambda x: 1.0, t=rt)
                for k in s:
                    s[k] = detect_distrib(s[k])
                tmp[rt]['detect'] = s
                s = {}
                s_d, tot_d = compute_scores(st, defender, dm, unit_types.shoot_down_set, unit_types.score_unit, t=rt)
                s_a, tot_a = compute_scores(st, attacker, dm, unit_types.ground_set, unit_types.score_unit, t=rt)
                for k in s_d:
                    s[k] = units_distrib(s_d[k] / (0.1 + s_a[k]))
                tmp[rt]['ground'] = s
                s = {}
                s_d, tot_d = compute_scores(st, defender, dm, unit_types.shoot_up_set, unit_types.score_unit, t=rt)
                s_a, tot_a = compute_scores(st, attacker, dm, unit_types.flying_set, unit_types.score_unit, t=rt)
                for k in s_d:
                    s[k] = units_distrib(s_d[k] / (0.1 + s_a[k]))
                tmp[rt]['air'] = s
            tests.append(tmp)
    return tests

class TacticalModel:
    """
    For all region r we have:
        A (Attack) in true/false
        EI (Economical importance) in {0, 1, 2} for the player considered
        TI (Tactical importance) in {0, 1, 2, 3, 4} for the player considered
        B (Belongs) in {True/False} for the player considered
            # P(A, EI, TI, B) = P(EI|A)P(TI|A)P(B|A)P(A)
        ==> P(A, EI, TI, B) = P(EI)P(TI)P(B)P(A | EI, TI, B)
        ?: P([A=1] | EI, TI) = sum_B[P([A=1] | EI, TI, B).P(B)]
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

    @staticmethod
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

    def train(self, battles):
        """
        fills Atrue_knowing_EI_TI_B and H_knowing_AD_GD_ID according to battles
        """
        sA = 0.0
        sH = 0.0
        for b in battles:
            #print b
            for keco,veco in b[1]['eco'].iteritems():
                for ktac,vtac in b[1]['tactic'].iteritems():
                    for kbel,vbel in b[1]['belong'].iteritems():
                        tmp = veco*vtac*vbel
                        self.Atrue_knowing_EI_TI_B[keco, ktac, kbel] += tmp 
                        sA += tmp
            for attack_type in b[0]:
                for kair,vair in b[1]['air'].iteritems():
                    for kground,vground in b[1]['ground'].iteritems():
                        for kdetect,vdetect in b[1]['detect'].iteritems():
                            tmp = vair*vground*vdetect
                            self.H_knowing_AD_GD_ID[TacticalModel.attack_type_to_ind(attack_type), kair, kground, kdetect] += tmp 
                            sH += tmp
        self.Atrue_knowing_EI_TI_B /= sA
        self.H_knowing_AD_GD_ID /= sH
        print "I've seen", len(battles), "battles"

    def test(self, tests, results):
        good_where = {'Reg': 0, 'CDR': 0}
        good_how = {'Reg': 0, 'CDR': 0}
        for i,t in enumerate(tests):
            for rt in t: # region types
                probabilities_where = {}
                max_where = -1.0
                where = 0
                for r in t[rt]['eco']:
                    tmp_prob = 0.0
                    for es in t[rt]['eco'][r]:
                        for ts in t[rt]['tactic'][r]:
                            for bs in t[rt]['belong'][r]:
                                tmp_prob += t[rt]['eco'][r][es] \
                                        * t[rt]['tactic'][r][ts] \
                                        * t[rt]['belong'][r][bs] \
                                        * self.Atrue_knowing_EI_TI_B[es,ts,bs]
                    probabilities_where[r] = tmp_prob
                    if tmp_prob > max_where:
                        max_where = tmp_prob
                        where = r
                probabilities_how = {}
                probabilities_where_how = {}
                max_where_how = -1.0
                how = 0
                where_how = 0
                for r in t[rt]['air']:
                    tmp_H_dist = [0.0, 0.0, 0.0, 0.0]
                    for ais in t[rt]['air'][r]:
                        for gs in t[rt]['ground'][r]:
                            for ds in t[rt]['detect'][r]:
                                tmp_H_dist += t[rt]['air'][r][ais] \
                                        * t[rt]['ground'][r][gs] \
                                        * t[rt]['detect'][r][ds] \
                                        * self.H_knowing_AD_GD_ID[:,ais,gs,ds]
                    probabilities_how[r] = tmp_H_dist
                    tmp = tmp_H_dist * probabilities_where[r]
                    for h,prob in enumerate(tmp):
                        if prob > max_where_how:
                            max_where_how = prob
                            how = h
                            where_how = r
                    probabilities_where_how[r] = tmp
                #print "Where (absolute):", where
                #print "Where|how:", where_how
                #print "How:", how
                for attack_type in results[i][0]:
                    if TacticalModel.attack_type_to_ind(attack_type) == how:
                        good_how[rt] += 1
                if results[i][-1][rt] == where_how:
                    good_where[rt] += 1
            #print results[i] # real battle that happened
        for rt in good_where:
            print "Type:", rt
            print "Good where predictions:", good_where[rt]*1.0/len(results)
            print "Good how predictions:", good_how[rt]*1.0/len(results)


if __name__ == "__main__":
    # serialize?
    fnamelist = []
    if sys.argv[1] == '-d':
        import glob
        fnamelist = glob.iglob(sys.argv[2] + '/*.rgd')
    else:
        fnamelist = sys.argv[1:]
    battles = []
    tests = []
    results = []
    if testing:
        i = 0
        learngames = []
        testgames = []
        for fname in fnamelist:
            i += 1
            if i > NUMBER_OF_TEST_GAMES:
                learngames.append(fname)
            else:
                testgames.append(fname)
        for fname in learngames:
            f = open(fname)
            floc = open(fname[:-3]+'rld')
            dm = DistancesMaps(floc)
            floc.close()
            floc = open(fname[:-3]+'rld')
            print "training on:", fname
            pm = PositionMapper(floc)
            players_races = data_tools.players_races(f)
            battles.extend(extract_tactics_battles(fname, dm, pm))
        for fname in testgames:
            f = open(fname)
            floc = open(fname[:-3]+'rld')
            dm = DistancesMaps(floc)
            floc.close()
            floc = open(fname[:-3]+'rld')
            print "testing on:", fname
            pm = PositionMapper(floc)
            players_races = data_tools.players_races(f)
            tests.extend(extract_tests(fname, dm, pm))
            results.extend(extract_tactics_battles(fname, dm, pm))
    else:
        for fname in fnamelist:
            f = open(fname)
            floc = open(fname[:-3]+'rld')
            dm = DistancesMaps(floc)
            floc.close()
            floc = open(fname[:-3]+'rld')
            print fname
            pm = PositionMapper(floc)
            players_races = data_tools.players_races(f)
            battles.extend(extract_tactics_battles(fname, dm, pm))
    tactics = TacticalModel()
    tactics.train(battles)
    if testing:
        tactics.test(tests, results)
    ### print tactics

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


