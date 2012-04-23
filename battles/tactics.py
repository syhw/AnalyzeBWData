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

DEBUG_LEVEL = 1 # 0: no debug output, 1: some, 2: all
HISTOGRAMS = True
testing = True # learn only or test on NUMBER_OF_TEST_GAMES
NUMBER_OF_TEST_GAMES = 10 # number of games to evaluates the tactical model
# if this number is greater than the total number of games,
# the test set will be the training set (/!\ BAD evaluation)

# TODO maxi refactor
# TODO Fix the tactical score
# TODO Try an eco score like the tactical score ==> sum to 1
# TODO verify probas (sum to 1) and tables contents (bias? priors?)
# TODO A (where happens the attack) comes from a distrib "where it is possible"
# TODO new evaluation metrics

SECONDS_BEFORE = 0 # number of seconds before the attack to update state
ADD_SMOOTH = 0.0 # Laplace smoothing, could be less than 1.0
TACT_PARAM = -1.6 # power of the distance of units to/from regions
# 1.6 means than a region which is at distance 1 of the two halves of the army
# of the player is 1.5 more important than one at distance 2 of the full army
WITH_DROP = True # with or without Drop as an attack type
INFER_DROP = True # with or without inference of drops
if not WITH_DROP:
    INFER_DROP = False
ALT_ECO_SCORE = False # compute an alternative eco score like the tactical one
ECO_SCORE_PARA = 1.6 # power of the distance of workers to/from regions
SOFT_EVIDENCE_AD_GD = False # tells if we should use binary AD and GD
bins_ad_gd = [0.0, 0.1, 0.5] # tells the bins lower limits (quantiles) values
if SOFT_EVIDENCE_AD_GD:
    bins_ad_gd = [0.0, 1.0]
bins_detect = [0.0, 0.99, 1.99] # none, one, many detectors
bins_tactical = [0.0, 0.1, 0.2, 0.4]
bins_eco = [0.0, 0.05, 0.51] # no eco, small eco, more than half of total
tactical_values = {'Reg': [], 'CDR': []}

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
    """ computes the scores with the scoring_f function of all regions 
    of type t and returns (scores_dict, total) """
    s = {}
    tot = 0.00000000001
    x = select(state, player, inset)
    for tmpr in dm.list_regions(t): # yes, it's dumb, but I don't want a sparse dict()
        s[tmpr] = 0.0               # even though I could get(x, default)...
        for unit in x:
            if unit[t] == tmpr:
                s[tmpr] += scoring_f(unit.name)
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
            if d >= 0.0:
                s[tmpr] += unit_types.score_unit(unit.name)*((1.0+d)**TACT_PARAM)
            else:
                s[tmpr] += unit_types.score_unit(unit.name)*(dm.max_dist**TACT_PARAM)
        tot += s[tmpr]
    #if HISTOGRAMS:
    #    for r,v in s.iteritems():
    #        tactical_values[t].append(v/tot)
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
    return s[r]/tot

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
    d = {}
    for i in range(len(bins_tactical)): 
        d[i] = 0.0
    d[where_bins(score, bins_tactical)] = 1.0
    return d

def eco_distrib(score):
    # TODO revise distrib into true distrib?
    d = {0: 0.0, 1: 0.0, 2: 0.0}
    d[where_bins(score, bins_eco)] = 1.0
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
    d = {0: 0.0, 1: 0.0, 2: 0.0}
    d[where_bins(score, bins_detect)] = 1.0
    return d

def units_distrib(score): # score is given relative to attackers force
    # TODO revise distrib into true distrib?
    #small defense (up to 2x smaller than attacker's force), >2x => big defense
    if SOFT_EVIDENCE_AD_GD:
        d = {0: 0.0, 1: 0.0}
        if score == 0.0:
            d[0] = 1.0
        elif score >= 1.0:
            d[1] = 1.0
        else:
            d[0] = 1.0 - score
            d[1] = score
        return d
    else:
        d = {};
        for i in range(len(bins_ad_gd)): 
            d[i] = 0.0
        d[where_bins(score, bins_ad_gd)] = 1.0
        return d


def detect_attacker(defender, d):
    if len(d) < 2:
        print "only one player in an attack"
        return -1
    for k in d:
        if k != defender:
            return k

def extract_tactics_battles(fname, pr, dm, pm=None):
    """ 
    Extract all attacks and tactics from the file named fname,
    with the distance map (between regions) dm, and the positional mapper pm
    returns a list of battles [([types], scoredict, cdr, reg)]
    """
    obs = attack_tools.Observers()
    st = state_tools.GameState()
    st.track_loc(open(fname[:-3]+'rld'))
    battles = []
    buf_lines = []
    started = False
    f = open(fname)
    for line in f:
        line = line.rstrip('\r\n')
        obs.detect_observers(line)
        l = line.split(',')
        if started and len(l) > 1:
            time_sec = int(l[0])/24
            buf_lines.append((time_sec, line))
            while len(buf_lines) and buf_lines[0][0] + SECONDS_BEFORE <= time_sec:
                st.update(buf_lines.pop(0)[1])
        else:
            st.update(line)
        if 'Begin replay data:' in line:
            started = True
        if 'IsAttacked' in line:
            tmpres = data_tools.parse_attacks(line)
            cdr = pm.get_CDR(tmpres[1][0], tmpres[1][1])
            reg = pm.get_Reg(tmpres[1][0], tmpres[1][1])
            units = data_tools.parse_dicts(line)
            units = obs.heuristics_remove_observers(units)
            if len(units[0]) < 2:
                continue
            defender = line.split(',')[1]
            attacker = detect_attacker(defender, units[0])
            if INFER_DROP and max([True if d in units[0][attacker] else False for d in unit_types.drop]):
                tmpres[0].append('DropAttack')

            tmp = {'Reg': {}, 'CDR': {}}
            for rt in tmp:
                s, tot = compute_scores(st, defender, dm, unit_types.workers, lambda x: 1.0, t=rt)
                for k in s:
                    s[k] = eco_distrib(s[k]/tot)
                tmp[rt]['eco'] = s
                s, tot = compute_tactical_scores(st, defender, dm, t=rt)
                #if HISTOGRAMS:
                #    if rt == 'Reg':
                #        tactical_values[rt].append(s[reg]/tot)
                #    else:
                #        tactical_values[rt].append(s[cdr]/tot)
                for k in s:
                    s[k] = tactic_distrib(s[k]/tot)
                tmp[rt]['tactic'] = s
                s, tot = compute_tactical_scores(st, attacker, dm, t=rt)
                if HISTOGRAMS:
                    if rt == 'Reg':
                        tactical_values[rt].append(s[reg]/tot)
                    else:
                        tactical_values[rt].append(s[cdr]/tot)
                for k in s:
                    s[k] = tactic_distrib(s[k]/tot)
                tmp[rt]['atactic'] = s
                tmp[rt]['belong'] = {}
                for r in dm.list_regions(rt):
                    tmp[rt]['belong'][r] = belong_distrib(r, defender, attacker, st, dm, t=rt)
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

            if DEBUG_LEVEL > 1:
                print tmpres[0]
                for rt in ['Reg', 'CDR']:
                    for k in tmp[rt]:
                        print k,
                        if rt == 'Reg':
                            print "DEBUG", tmp[rt][k][reg]
                        else:
                            print "DEBUG", tmp[rt][k][cdr]

            battles.append((tmpres[0], tmp, pr[attacker], {'Reg': reg, "CDR": cdr}))
            #              (list of types, dict of distribs, CDR, Reg)
    return battles

def extract_tests(fname, dm, pm=None):
    """
    Extract tests situations from the file named fname,
    with the distance map (between regions) dm, and the positional mapper pm
    returns a list of dicts of regions scores:
    [{'Reg': {score_type: {reg: distrib}},'CDR': {score_type: {reg: distrib}}}]
    """
    obs = attack_tools.Observers()
    st = state_tools.GameState()
    st.track_loc(open(fname[:-3]+'rld'))
    tests = []
    buf_lines = []
    started = False
    f = open(fname)
    for line in f:
        line = line.rstrip('\r\n')
        obs.detect_observers(line)
        l = line.split(',')
        if started and len(l) > 1:
            time_sec = int(l[0])/24
            buf_lines.append((time_sec, line))
            while len(buf_lines) and (buf_lines[0][0] + SECONDS_BEFORE) <= time_sec:
                st.update(buf_lines.pop(0)[1])
        else:
            st.update(line)
        if 'Begin replay data:' in line:
            started = True
        if 'IsAttacked' in line:
            units = data_tools.parse_dicts(line)
            units = obs.heuristics_remove_observers(units)
            if len(units[0]) < 2:
                continue
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
                s, tot = compute_tactical_scores(st, attacker, dm, t=rt)
                for k in s:
                    s[k] = tactic_distrib(s[k]/tot)
                tmp[rt]['atactic'] = s
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

class TacticsMatchUp:
    """
    Regroups all the tactical models for each races in a given match-up
    interface by count_battles (dispatch to the right models) and normalize()
    """

    def __init__(self):
        self.models = {'P': TacticalModel(), 'T': TacticalModel(), 'Z': TacticalModel()}

    def __repr__(self):
        s = ""
        for k,v in self.models.iteritems():
            if v.n_battles['Reg'] <= 0.0 or v.n_battles['CDR'] <= 0.0:
                continue
            s += "**************** \n"
            s += "Model for race " + k + "\n"
            s += "**************** \n"
            s += v.__repr__() + '\n'
        return s

    def count_battles(self, battles):
        for b in battles:
            for rt in ['Reg', 'CDR']:
                self.models[b[-2]].count_battle(b, rt)

    def normalize(self):
        for k,m in self.models.iteritems():
            print "Model race", k
            m.normalize()

    def test(self, tests, results):
        td = {'P': [], 'T': [], 'Z': []}
        rd = {'P': [], 'T': [], 'Z': []}
        for i,r in enumerate(results):
            rd[r[-2]].append(r)
            td[r[-2]].append(tests[i])
        for k in td:
            self.models[k].test(td[k], rd[k])

    def plot_tables(self):
        import matplotlib.pyplot as plt

        for race,m in self.models.iteritems():
            if m.n_battles['Reg'] <= 0.0 or m.n_battles['CDR'] <= 0.0:
                continue
            for rt,t in m.EI_TI_B_ATI_knowing_A.iteritems():
                if DEBUG_LEVEL > 1:
                    print "Total sum P(EI,TI,B,ATI|A)", [sum(sum(sum(sum(t[:,:,:,i])))) for i in range(2)]
                fig = plt.figure()
                fig.subplots_adjust(wspace=0.3, hspace=0.6)

                ax = fig.add_subplot(221)
                ind = np.arange(len(bins_eco))
                width = 0.5
                ax.set_xticks(ind+width)
                ax.set_xticklabels(["no eco", "low eco", "high eco"])
                s = [sum(sum(sum(t[i,:,:,:,1]))) for i in ind]
                #print s
                ax.bar(ind, s, width, color='r')

                ax = fig.add_subplot(222)
                ind = np.arange(len(bins_tactical))
                ax.set_xticks(ind+width)
                ax.set_xlabel("tactical value")
                ax.set_xticklabels(["0", "1", "2", "3", "4"])
                s = [sum(sum(sum(t[:,i,:,:,1]))) for i in ind]
                #print s
                ax.bar(ind, s, width, color='r')

                ax = fig.add_subplot(223)
                ind = np.arange(2)
                ax.set_xticks(ind+width)
                ax.set_xticklabels(["doesn't belong", "belong"])
                s = [sum(sum(sum(t[:,:,i,:,1]))) for i in ind]
                #print s
                ax.bar(ind, s, width, color='r')
                
                #plt.show()
                plt.savefig("where"+rt+race+".png")

            for rt,t in m.AD_GD_ID_knowing_H.iteritems():
                if DEBUG_LEVEL > 1:
                    print "Total sum P(AD,GD,ID|H)", [sum(sum(sum(t[:,:,:,i]))) for i in range(len(t[0,0,0]))]
                fig = plt.figure()
                fig.subplots_adjust(wspace=0.3, hspace=0.6)

                ax = fig.add_subplot(321)
                ind = np.arange(len(bins_ad_gd))
                ax.set_xticks(ind+width)
                ax.set_xlabel("air defense level")
                ax.set_ylabel("P(Air)")
                #ax.set_xticklabels(["", "", ""])
                s = [sum(sum(t[i,:,:,1])) for i in ind]
                #print s
                ax.bar(ind, s, width, color='r')

                ax = fig.add_subplot(322)
                ind = np.arange(len(bins_ad_gd))
                ax.set_xticks(ind+width)
                ax.set_xlabel("ground defense level")
                ax.set_ylabel("P(Ground)")
                s = [sum(sum(t[:,i,:,0])) for i in ind]
                #print s
                ax.bar(ind, s, width, color='r')

                ax = fig.add_subplot(323)
                ind = np.arange(len(bins_ad_gd))
                ax.set_xticks(ind+width)
                ax.set_xlabel("ground defense level")
                ax.set_ylabel("P(Invis)")
                s = [sum(sum(t[:,i,:,2])) for i in ind]
                #print s
                ax.bar(ind, s, width, color='r')

                ax = fig.add_subplot(324)
                ind = np.arange(len(bins_detect))
                ax.set_xticks(ind+width)
                ax.set_xlabel("invis defense level")
                ax.set_ylabel("P(Invis)")
                s = [sum(sum(t[:,:,i,2])) for i in ind]
                #print s
                ax.bar(ind, s, width, color='r')

                if WITH_DROP:
                    ax = fig.add_subplot(325)
                    ind = np.arange(len(bins_ad_gd))
                    ax.set_xticks(ind+width)
                    ax.set_xlabel("air defense level")
                    ax.set_ylabel("P(Drop)")
                    s = [sum(sum(t[i,:,:,3])) for i in ind]
                    #print s
                    ax.bar(ind, s, width, color='r')

                    ax = fig.add_subplot(326)
                    ind = np.arange(len(bins_ad_gd))
                    ax.set_xticks(ind+width)
                    ax.set_xlabel("ground defense level")
                    ax.set_ylabel("P(Drop)")
                    s = [sum(sum(t[:,i,:,3])) for i in ind]
                    #print s
                    ax.bar(ind, s, width, color='r')

                #plt.show()
                plt.savefig("how"+rt+race+".png")


class TacticalModel:
    """
    For all region r we have:
        A (Attack) in true/false
        EI (Economical importance) in {0, 1, 2} for the player considered
        TI (Tactical importance) in {0, 1, 2, 3, 4} for the player considered
        B (Belongs) in {True/False} for the player considered
        ==> P(A,EI,TI,B) = P(A)P(EI,TI,B|A)
        ?: P(A|EI,TI,P(B')) = sum_B[P(B').P(lambda|B,B').P(EI,TI,B|A)].P(A)
        P(B=True) = 1.0 iff r si one of the bases of the player considered
        P(B=False) = 1.0 iff r si one of the bases of the ennemy of the player
        P(B=True) prop to min_{base \in players'bases}(dist(r, base))
        P(lambda|B,B') = 1.0 ssi B == B'
        ex.: P(B=True) = 0.5 in the middle of the map

        H (How) in {Ground, Air, Invisible, Drop}
            WITH_DROP = True => H has "Drop", otherwise not
        AD (Air defense) in {0, 1, 2} (0: no defense, 1: light defense compared
        GD (Ground defense) in {0, 1, 2}  ..to the attacker, 2: heavy defense)
        ID (Invisible defense = #detectors) in {0, 1, 2+}
        ==> P(H,AD,GD,ID) = P(AD,GD,ID|H).P(H)
        ?: P(H|AD,GD,ID) = P(AD,GD,ID|H).P(H)/P(AD,GD,ID)
        P(A/G/ID=0) = 1.0 iff r has no defense against this type of attack
        P(A/G/ID=1) = 1.0 iff r has less than one half the score of the assaillant
                              on a given attack type
        P(A/G/ID=2) = 1.0 iff r has more than one half the score of the assaillant
                              on a given attack type
    """

    def __init__(self):
        # EI_TI_B_ATI_knowing_A[rt][ei][ti][b] = P(ei,ti,b | A[=0,1])
        # AD_GD_ID_knowing_H[rt][ad][gd][id] = P(ad,gd,id | H[=ground/air/drop/invis])
        self.EI_TI_B_ATI_knowing_A = {}
        self.AD_GD_ID_knowing_H = {}
        self.size_H = 3
        if WITH_DROP:
            self.size_H = 4
        for rt in ['Reg', 'CDR']:
            self.EI_TI_B_ATI_knowing_A[rt] = np.ndarray(shape=(len(bins_eco),len(bins_tactical),2,len(bins_tactical),2), dtype='float')
            self.EI_TI_B_ATI_knowing_A[rt].fill(ADD_SMOOTH)
            self.AD_GD_ID_knowing_H[rt] = np.ndarray(shape=(len(bins_ad_gd),len(bins_ad_gd),len(bins_detect),self.size_H), dtype='float')
            self.AD_GD_ID_knowing_H[rt].fill(ADD_SMOOTH)
        self.n_battles = {'Reg': 0.0, 'CDR': 0.0}
        self.n_not_battles = {'Reg': 0.0, 'CDR': 0.0}
        self.A = {'Reg': 0.5, 'CDR': 0.5}
        self.n_how = {'Reg': [0.0 for i in range(self.size_H)], 'CDR': [0.0 for i in range(self.size_H)]}
        self.H = {'Reg': [1.0/self.size_H for i in range(self.size_H)], 
                'CDR': [1.0/self.size_H for i in range(self.size_H)]}

    def __repr__(self):
        s = ""
        for rt in ['Reg', 'CDR']:
            s += "\nRegion type" + rt + "\n"
            s += "*** P(EI, TI, B | A) ***\n"
            s += self.EI_TI_B_ATI_knowing_A[rt].__repr__() + '\n'
            s += "*** P(AD, GD, ID | H) ***\n"
            s += self.AD_GD_ID_knowing_H[rt].__repr__() + '\n'
        return s

    @staticmethod
    def attack_type_to_ind(at):
        if at == 'GroundAttack':
            return 0
        elif at == 'AirAttack':
            return 1
        elif at == 'InvisAttack':
            return 2
        elif WITH_DROP and at == 'DropAttack':
            return 3
        else:
            print "Not a good attack type label"
            raise TypeError
    
    def count_battle(self, b, rt):
        """
        fills EI_TI_B_ATI_knowing_A[rt] and AD_GD_ID_knowing_H[rt] according to b
        """
        rnumber = b[-1][rt]
        for keco,veco in b[1][rt]['eco'][rnumber].iteritems():
            for ktac,vtac in b[1][rt]['tactic'][rnumber].iteritems():
                for kbel,vbel in b[1][rt]['belong'][rnumber].iteritems():
                    for katac,vatac in b[1][rt]['atactic'][rnumber].iteritems():
                        tmp = veco*vtac*vbel*vatac
                        self.EI_TI_B_ATI_knowing_A[rt][keco, ktac, kbel, katac, 1] += tmp 
                        self.n_battles[rt] += tmp
        for r in b[1][rt]['eco']:
            if r != rnumber:
                for keco,veco in b[1][rt]['eco'][r].iteritems():
                    for ktac,vtac in b[1][rt]['tactic'][r].iteritems():
                        for kbel,vbel in b[1][rt]['belong'][r].iteritems():
                            for katac,vatac in b[1][rt]['atactic'][r].iteritems():
                                tmp = veco*vtac*vbel*vatac
                                self.EI_TI_B_ATI_knowing_A[rt][keco, ktac, kbel, katac, 0] += tmp 
                                self.n_not_battles[rt] += tmp

        for attack_type in b[0]:
            for kair,vair in b[1][rt]['air'][rnumber].iteritems():
                for kground,vground in b[1][rt]['ground'][rnumber].iteritems():
                    for kdetect,vdetect in b[1][rt]['detect'][rnumber].iteritems():
                        tmp = vair*vground*vdetect
                        ind = TacticalModel.attack_type_to_ind(attack_type)
                        self.AD_GD_ID_knowing_H[rt][kair, kground, kdetect,ind] += tmp
                        self.n_how[rt][ind] += tmp
 
    def normalize(self):
        if self.n_battles['Reg'] <= 0.0 or self.n_battles['CDR'] <= 0.0:
            return
        for rt in self.AD_GD_ID_knowing_H:
            self.A[rt] = self.n_battles[rt]/(self.n_battles[rt] + self.n_not_battles[rt])
            self.H[rt] = [self.n_how[rt][i]/sum(self.n_how[rt]) for i in range(self.size_H)]
            for ind in range(len(self.n_how[rt])):
                self.AD_GD_ID_knowing_H[rt][:,:,:,ind] /= self.n_how[rt][ind] + len(self.AD_GD_ID_knowing_H[rt])*len(self.AD_GD_ID_knowing_H[rt][0])*len(self.AD_GD_ID_knowing_H[rt][0][0])*ADD_SMOOTH
            #self.EI_TI_B_ATI_knowing_A[rt][:,:,:,:,1] /= sum(sum(sum(sum(self.EI_TI_B_ATI_knowing_A[rt][:,:,:,:,1]))))
            self.EI_TI_B_ATI_knowing_A[rt][:,:,:,:,1] /= len(self.EI_TI_B_ATI_knowing_A[rt])*len(self.EI_TI_B_ATI_knowing_A[rt][0])*len(self.EI_TI_B_ATI_knowing_A[rt][0][0])*len(self.EI_TI_B_ATI_knowing_A[rt][0][0][0])*ADD_SMOOTH + self.n_battles[rt]
            #self.EI_TI_B_ATI_knowing_A[rt][:,:,:,:,0] /= sum(sum(sum(sum(self.EI_TI_B_ATI_knowing_A[rt][:,:,:,:,0]))))
            self.EI_TI_B_ATI_knowing_A[rt][:,:,:,:,0] /= len(self.EI_TI_B_ATI_knowing_A[rt])*len(self.EI_TI_B_ATI_knowing_A[rt][0])*len(self.EI_TI_B_ATI_knowing_A[rt][0][0])*len(self.EI_TI_B_ATI_knowing_A[rt][0][0][0])*ADD_SMOOTH + self.n_not_battles[rt]
        print "I've seen", int(self.n_battles['Reg']), "Reg battles" # n_battles counted
        print "I've seen", int(self.n_battles['CDR']), "CDR battles" # n_battles counted
        # twice (both for CDR and Reg), n_how too

    def ask_A(self, rt='Reg', EI=-1, TI=-1, B=-1, ATI=-1, A=1):
        """ 
        returns the P(A|EI,TI,B) with given values, for all -1 values,
        it just sums on it (for instance, B=-1 means:
        P(A|EI,TI) = \sum_B[P(EI,TI,B|A).P(A)] / \sum_{A,B}[P(EI,TI,B|A).P(A)]
        """
        t = self.EI_TI_B_ATI_knowing_A[rt]
        P_A = [1.0 - self.A[rt], self.A[rt]]
        if EI == -1:
            t = sum(t)
        else:
            t = t[EI]
        if TI == -1:
            t = sum(t)
        else:
            t = t[TI]
        if B == -1:
            t = sum(t)
        else:
            t = t[B]
        if ATI == -1:
            t = sum(t)
        else:
            t = t[ATI]
        if A == -1:
            return np.array([t[i]*P_A[i] for i in range(2)]) / sum([t[i]*P_A[i] for i in range(2)])
        else:
            return t[A]*P_A[A] / sum([t[i]*P_A[i] for i in range(2)])

    def ask_H(self, rt='Reg', AD=-1, GD=-1, ID=-1, H=-1):
        """ 
        returns the P(H|AD,GD,ID) with given values, for all -1 values,
        it just sums on it (c.f. ask_A above)
        P(H|AD,GD,ID) = \sum_{AD,GD,ID}[P(AD,GD,ID|H).P(H)] 
                / \sum_{H,AD,GD,ID}[P(AD,GD,ID|H).P(H)]
        """
        t = self.AD_GD_ID_knowing_H[rt]
        P_H = self.H[rt]
        if AD == -1:
            t = sum(t)
        else:
            t = t[AD]
        if GD == -1:
            t = sum(t)
        else:
            t = t[GD]
        if ID == -1:
            t = sum(t)
        else:
            t = t[ID]
        if H == -1:
            return np.array([t[i]*P_H[i] for i in range(self.size_H)]) / sum([t[i]*P_H[i] for i in range(self.size_H)])

        else:
            return t[H]*P_H[H] / sum([t[i]*P_H[i] for i in range(self.size_H)])

    def test(self, tests, results):
        if len(results) == 0:
            return 
        good_where_how = {'Reg': 0, 'CDR': 0}
        good_where = {'Reg': 0, 'CDR': 0}
        number_at = {} # number of attacks for each attack type
        good_how = {'Reg': {}, 'CDR': {}}
        rank_where = {'Reg': 0, 'CDR': 0}
        proba_where = {'Reg': 0.0, 'CDR': 0.0}
        distance_where = {'Reg': 0.0, 'CDR': 0.0}
        for i,t in enumerate(tests):
            for rt in t: # region types
                probabilities_where = {}
                max_where = -1.0
                where = 0
                for r in t[rt]['eco']:
                    tmp_num = 0.0
                    tmp_denum = 0.0
                    for es in t[rt]['eco'][r]:
                        for ts in t[rt]['tactic'][r]:
                            for bs in t[rt]['belong'][r]:
                                for ats in t[rt]['atactic'][r]:
                                    tmp_num += t[rt]['eco'][r][es] \
                                            * t[rt]['tactic'][r][ts] \
                                            * t[rt]['belong'][r][bs] \
                                            * self.EI_TI_B_ATI_knowing_A[rt][es,ts,bs,ats,1]
                                    tmp_denum += t[rt]['eco'][r][es] \
                                            * t[rt]['tactic'][r][ts] \
                                            * t[rt]['belong'][r][bs] \
                                            * self.EI_TI_B_ATI_knowing_A[rt][es,ts,bs,ats,0]
                    probabilities_where[r] = tmp_num/(tmp_num+tmp_denum)
                    if probabilities_where[r] > max_where:
                        max_where = probabilities_where[r]
                        where = r
                probabilities_how = {}
                probabilities_where_how = {}
                max_where_how = -1.0
                how = 0
                where_how = 0
                for r in t[rt]['air']:
                    tmp_H_dist = [0.0, 0.0, 0.0] # w/o drop
                    if WITH_DROP:
                        tmp_H_dist = [0.0, 0.0, 0.0, 0.0]
                    for ais in t[rt]['air'][r]:
                        for gs in t[rt]['ground'][r]:
                            for ds in t[rt]['detect'][r]:
                                tmp_H_dist += t[rt]['air'][r][ais] \
                                        * t[rt]['ground'][r][gs] \
                                        * t[rt]['detect'][r][ds] \
                                        * self.AD_GD_ID_knowing_H[rt][ais,gs,ds,:]
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
                    number_at[attack_type] = number_at.get(attack_type, 0) + 1
                    if TacticalModel.attack_type_to_ind(attack_type) == how:
                        good_how[rt][attack_type] = good_how[rt].get(attack_type, 0) + 1
                    #if len(results[i][0]) > 1:
                    #    probabilities_how[where] 
                if results[i][-1][rt] == where_how:
                    good_where_how[rt] += 1
                if results[i][-1][rt] == where:
                    good_where[rt] += 1
            #print results[i] # real battle that happened
        for rt in good_where_how:
            print "Type:", rt
            print "Good where predictions:", good_where[rt]*1.0/len(results)
            print "Good where+how predictions:", good_where_how[rt]*1.0/len(results)
            total = 0
            good = 0
            for attack_type in number_at:
                total += number_at[attack_type]
                gh = good_how[rt].get(attack_type, 0)
                good += gh
                print "Good how", attack_type, "predictions:", gh*1.0/(number_at[attack_type]/2), ":", gh, "/", number_at[attack_type]/2
            print "Good how predictions:", good*1.0/(total/2), good, "/", total/2


if __name__ == "__main__":
    # serialize?
    fnamelist = []
    if sys.argv[1] == '-d':
        import glob
        for g in glob.iglob(sys.argv[2] + '/*.rgd'):
            fnamelist.append(g)
    else:
        fnamelist = sys.argv[1:]
    tests = []
    results = []
    tactics = TacticsMatchUp()
    learn = True
    if '-s' in sys.argv[1:]:
        learn = False
    if testing:
        i = 0
        learngames = []
        testgames = []
        if NUMBER_OF_TEST_GAMES < len(fnamelist):
            for fname in fnamelist:
                i += 1
                if i > NUMBER_OF_TEST_GAMES:
                    learngames.append(fname)
                else:
                    testgames.append(fname)
        else:
            print >> sys.stderr, "Number of test games > number of games"
            print >> sys.stderr, "Test and train on the same (whole set) games"
            learngames = fnamelist
            testgames = fnamelist

        # Learning
        if learn == False:
            try:
                tactics = pickle.load(open('models.pickle', 'r'))
            except:
                learn = True
        if learn == True:
            for fname in learngames:
                f = open(fname)
                floc = open(fname[:-3]+'rld')
                dm = DistancesMaps(floc)
                floc.close()
                print >> sys.stderr, "training on:", fname
                pm = PositionMapper(dm, fname[:-3])
                pr = data_tools.players_races(f)
                tactics.count_battles(extract_tactics_battles(fname, pr, dm, pm))

        # Testing
        for fname in testgames:
            f = open(fname)
            floc = open(fname[:-3]+'rld')
            dm = DistancesMaps(floc)
            floc.close()
            print >> sys.stderr, "testing on:", fname
            pm = PositionMapper(dm, fname[:-3])
            pr = data_tools.players_races(f)
            tests.extend(extract_tests(fname, dm, pm))
            results.extend(extract_tactics_battles(fname, pr, dm, pm))
    else:
        # Learning only
        if learn == False:
            try:
                tactics = pickle.load(open('models.pickle', 'r'))
            except:
                learn = True
        if learn == True:
            for fname in fnamelist:
                f = open(fname)
                floc = open(fname[:-3]+'rld')
                dm = DistancesMaps(floc)
                floc.close()
                print >> sys.stderr, fname
                pm = PositionMapper(dm, fname[:-3])
                pr = data_tools.players_races(f)
                tactics.count_battles(extract_tactics_battles(fname, pr, dm, pm))

    tactics.normalize()
    if '-s' in sys.argv[1:]:
        pickle.dump(tactics, open('models.pickle', 'w'))
    if testing:
        tactics.test(tests, results)
    if DEBUG_LEVEL > 0:
        print tactics
    if '-p' in sys.argv[1:]:
        tactics.plot_tables()

    if HISTOGRAMS:
        import matplotlib.pyplot as plt
        plt.figure()
        n, bins, patches = plt.hist(tactical_values['Reg'],5)
        plt.savefig("histTacticalReg.png")
        print bins
        plt.figure()
        n, bins, patches = plt.hist(tactical_values['CDR'],5)
        plt.savefig("histTacticalCDR.png")
        print bins
        plt.figure()
        tactical_values['Reg'].sort()
        t = tactical_values['Reg']
        bins = [0.0, t[int(len(t)*0.2)], t[int(len(t)*0.4)], t[int(len(t)*0.6)], t[int(len(t)*0.8)], 1.0]
        bins = bins_tactical
        print bins
        plt.hist(tactical_values['Reg'], bins)
        plt.savefig("myhistTacticalReg.png")
        plt.figure()
        tactical_values['CDR'].sort()
        t = tactical_values['CDR']
        bins = [0.0, t[int(len(t)*0.2)], t[int(len(t)*0.4)], t[int(len(t)*0.6)], t[int(len(t)*0.8)], 1.0]
        bins = bins_tactical
        print bins
        plt.hist(tactical_values['CDR'], bins)
        plt.savefig("myhistTacticalCDR.png")

    #P(A=1|EI=1,B=1) axes for values of TI
    #def ask_A(self, rt='Reg', EI=-1, TI=-1, B=-1, ATI=-1, A=1):
    t = tactics.models['T'].EI_TI_B_ATI_knowing_A['Reg']
    print [tactics.models['T'].ask_A('Reg', EI=1, TI=i, B=1, ATI=-1, A=1) for i in range(len(bins_tactical))]
    a = tactics.models['T'].A['Reg']
    num1 = np.array([sum(t[1,i,1,:,1])*a for i in range(len(bins_tactical))])
    num2 = np.array([sum(t[1,i,1,:,0])*(1.0-a) for i in range(len(bins_tactical))])
    print "P(A=1|EI=1,B=1) axes for values of TI"
    print num1/(num1+num2)

    
