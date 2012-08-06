#!/usr/bin/python
# -*- coding: utf-8 -*-

# Clustering to try:
#  - roll our own (see ArmyCompositions)
#  - PCA fragmentation on the first principal component
#  - PCA fragmentation on first few principal components
#  - Linear Discriminant Analysis same as above
#  - GMM EM (which distance function w.r.t. the 0 !!= 0.01 problem???)
#  - GMM DP (which distance function w.r.t. the 0 !!= 0.01 problem???)
#  - Latent Dirichlet Analysis TODO
# see SCALE_UP_SPECIAL_UNITS / scale_up_special (de-linearize)

import sys, os, pickle, copy, itertools, functools, math, random
import pylab as pl
from collections import defaultdict
from common import data_tools
from common import unit_types
from common import attack_tools
from common import state_tools
from common.vector_X import *
from common.common_tools import memoize
try:
    import numpy as np
except:
    print "you need numpy"
    sys.exit(-1)

SCORES_REGRESSION = False # try to do battles scores regressions
MIN_POP_ENGAGED = 6 # 12 zerglings, 6 marines, 3 zealots
MAX_FORCES_RATIO = 1.4 # max differences between engaged forces
WITH_STATIC_DEFENSE = False # tells if we include static defense in armies
WITH_WORKERS = False # tells if we include workers in armies
CSV_ARMIES_OUTPUT = True # CSV output of the armies compositions
DEBUG_OUR_CLUST = False # debugging output for our clustering
DEBUG_GMM = False # debugging output for Gaussian mixtures clustering
SHOW_NORMALIZE_OUTPUT = False # show normalized tables which are not uniform
SERIALIZE_GMM = True
NUMBER_OF_TEST_GAMES = 0 # number of test games to use
PARALLEL_COORDINATES_PLOT = False # should we plot units percentages?
SCALE_UP_SPECIAL_UNITS = False # scale up special units in the list of percents
ADD_SMOOTH_EC_EC = 0.01 # smoothing
LEARNED_EC_KNOWING_ETT = True # should we learn P(EC^{t+1}|ETT)? False not impl
WITH_SCORE_RATIO = True # use score ratio instead of just counting for units
WITH_STATE = True # use state, and so P(EC^{t+1}|TT) and P(EC|EC^{t+1})
SECONDS_BEFORE = 120 # number of seconds for between t and t+1
ADD_SMOOTH_EC_TT = 0.01
ADD_SMOOTH_C_EC = 0.01
STATIC_DEFENSE_MULTIPLIER = 1.5 # how much to multiply static defense score by
PLOT_EC_KNOWING_ECNEXT = True
PLOT_W_KNOWING_C_EC = True
PLOT_ECNEXT_KNOWING_ETT = False
disc_width = 0.01 # width of bins in P(Unit_i | C)
epsilon = 1.0e-6 # lowest not zero

print >> sys.stderr, "args", sys.argv[1:]
print >> sys.stderr, "SCORES_REGRESSION ",    SCORES_REGRESSION 
print >> sys.stderr, "MIN_POP_ENGAGED ",      MIN_POP_ENGAGED 
print >> sys.stderr, "MAX_FORCES_RATIO ",     MAX_FORCES_RATIO 
print >> sys.stderr, "WITH_STATIC_DEFENSE ",  WITH_STATIC_DEFENSE 
if WITH_STATIC_DEFENSE:
    print >> sys.stderr, "STATIC_DEFENSE_MULTIPLIER", STATIC_DEFENSE_MULTIPLIER
print >> sys.stderr, "WITH_WORKERS ",         WITH_WORKERS 
print >> sys.stderr, "CSV_ARMIES_OUTPUT ",    CSV_ARMIES_OUTPUT 
print >> sys.stderr, "DEBUG_OUR_CLUST ",      DEBUG_OUR_CLUST 
print >> sys.stderr, "DEBUG_GMM ",            DEBUG_GMM
print >> sys.stderr, "SHOW_NORMALIZE_OUTPUT ",SHOW_NORMALIZE_OUTPUT 
print >> sys.stderr, "NUMBER_OF_TEST_GAMES ", NUMBER_OF_TEST_GAMES 
print >> sys.stderr, "PARALLEL_COORDINATES_PLOT ", PARALLEL_COORDINATES_PLOT 
print >> sys.stderr, "SCALE_UP_SPECIAL_UNITS ", SCALE_UP_SPECIAL_UNITS 
print >> sys.stderr, "WITH_SCORE_RATIO",      WITH_SCORE_RATIO 
print >> sys.stderr, "WITH_STATE",            WITH_STATE 
print >> sys.stderr, "SECONDS_BEFORE",        SECONDS_BEFORE 


def evaluate_pop(d):
    r = {}
    for (k,v) in d.iteritems():
        s = 0
        for (kk, vv) in v.iteritems():
            s += unit_types.unit_double_pop.get(kk,0)*vv
        r[k] = s
    return r

def score_units(d):
    r = {}
    for k,v in d.iteritems():
        r[k] = 0
        for unit, numbers in v.iteritems():
            r[k] += unit_types.score_unit(unit)*numbers
    return r

def to_ratio(d):
    r = {}
    for k,v in d.iteritems():
        tmp = {}
        for unit, numbers in v.iteritems():
            tmp[unit] = 0.0
            if unit in unit_types.static_defense_set:
                if not WITH_STATIC_DEFENSE:
                    continue
                else:
                    tmp[unit] = STATIC_DEFENSE_MULTIPLIER
            if not WITH_WORKERS:
                if unit in unit_types.workers:
                    continue
            if tmp[unit] == 0.0: # we will count this unit
                tmp[unit] = 1.0  # but the normal multiplier (1.0)
            if WITH_SCORE_RATIO:
                tmp[unit] *= unit_types.score_unit(unit)*numbers
            else:
                tmp[unit] *= numbers
        s = sum(tmp.itervalues())
        for unit in tmp:
            tmp[unit] /= (s+disc_width) # +width to avoid 100% (99%)
        r[k] = tmp
    return r


def extract_armies_battles(f):
    """ take a file and parse it for attacks, returning a list of attacks
    with for each time of the beginning of the attack,
    the (max) armies of each player in form of a dict and
    the remaining units at the end in form of a dict and
    the number of workers lost for each player:
        [(attack_time, {plid:{uid:nb}}, {plid:{uid:nb}}, {plid:wrks_lost})] """
    attacks = []
    obs = attack_tools.Observers()
    if WITH_STATE:
        st = state_tools.GameState()
        buf_lines = []
        # the replay is already started by data_tools.players_races

    for line in f:
        line = line.rstrip('\r\n')
        obs.detect_observers(line)
        if WITH_STATE:
            l = line.split(',')
            if len(l) > 1:
                time_sec = int(l[0])/24
                buf_lines.append((time_sec, line))
                while len(buf_lines) and buf_lines[0][0] + SECONDS_BEFORE <= time_sec:
                    st.update(buf_lines.pop(0)[1])

        if 'IsAttacked' in line:
            tmp = data_tools.parse_dicts(line, lambda x: int(x))
            # sometimes the observers are detected in the fight (their SCVs)
            tmp = obs.heuristics_remove_observers(tmp)
            if len(tmp[0]) == 2: # when a player killed observer's units...
                if WITH_STATE:
                    buildings = {}
                    for pl in tmp[0]:
                        buildings[pl] = st.get_buildings(pl)
                    military_units = {}
                    for pl in tmp[0]:
                        military_units[pl] = st.get_military(pl)
                    attacks.append((int(line.split(',')[0]),tmp[0], tmp[1], tmp[2], buildings, military_units))
                else:
                    attacks.append((int(line.split(',')[0]),tmp[0], tmp[1], tmp[2]))
    return attacks

def format_battle_init(players_races, army):
    p = []
    for k in army.iterkeys():
        p.append((k,players_races[k]))
    assert(len(p) == 2)
    p1 = p[0][0]
    p2 = p[1][0]
    if p[0][1] == 'T' and p[1][1] == 'P':
        p1 = p[1][0]
        p2 = p[0][0]
    elif p[0][1] == 'Z':
        p1 = p[1][0]
        p2 = p[0][0]
    return p1, p2

def append_units_numbers(l, player, players_races, army):
    for unit in unit_types.by_race.military[players_races[player]]:
        l.append(army[player].get(unit, 0))
    for unit in unit_types.by_race.static_defense[players_races[player]]:
        l.append(army[player].get(unit, 0))
    l.append(army[player].get(unit_types.by_race.workers[
        players_races[player]], 0))

def format_battle_for_regr(players_races, armies_battle):
    """ take an "extract_armies_battles" formatted battle data and make it
    ready for regression with P > T > Z in order, then on player id order,
    and player1_score - player2_score """
    p1, p2 = format_battle_init(players_races, armies_battle[1])
    tmp = []
    append_units_numbers(tmp, p1, players_races, armies_battle[1])
    append_units_numbers(tmp, p2, players_races, armies_battle[1])
    #scores = evaluate_pop(armies_battle[2])
    scores = score_units(armies_battle[2])
    wrk_scores = armies_battle[3][p1]*2 - armies_battle[3][p1]*2
    tmp.append(scores[p1] - scores[p2] + wrk_scores)
    return tmp

def format_battle_for_clust_adv(players_races, armies_battle):
    """ take an "extract_armies_battles" formatted battle data and make it
    ready for clustering (order P > T > Z) by considering only battles 
    which were relevant (comparable units scores for both parties), 
    returning 2 vectors of units ratio (of total army) composition
    per unit types and the final scores of both players"""
    p1, p2 = format_battle_init(players_races, armies_battle[1])
    pop_max = evaluate_pop(armies_battle[1])
    #pop_after = evaluate_pop(armies_battle[2])
    score_before = score_units(armies_battle[1])
    races = {0: players_races[p1], 1: players_races[p2]}
    if pop_max[p1] > MIN_POP_ENGAGED*2 and pop_max[p2] > MIN_POP_ENGAGED*2 and score_before[p1] < MAX_FORCES_RATIO*score_before[p2] and score_before[p2] < MAX_FORCES_RATIO*score_before[p1]:
        compo = to_ratio(armies_battle[1])
        # these two loops just for Dark Archon's mind controlled armies
        for k in compo[p1]:
            if k[0] != players_races[p1]:
                if WITH_STATE:
                    return [], [], [], [], [], [], set(), set(), [], [], {}
                else:
                    return [], [], [], [], [], [], {}
        for k in compo[p2]:
            if k[0] != players_races[p2]:
                if WITH_STATE:
                    return [], [], [], [], [], [], set(), set(), [], [], {}
                else:
                    return [], [], [], [], [], [], {}
        # /these two loops just for Dark Archon's mind controlled armies
        score_after = score_units(armies_battle[2])
        if WITH_STATE:
            previous_compo = to_ratio(armies_battle[-1])
            return compo[p1], compo[p2], score_before[p1], score_before[p2], score_after[p1], score_after[p2], armies_battle[-2][p1], armies_battle[-2][p2], previous_compo[p1], previous_compo[p2], races
        else:
            return compo[p1], compo[p2], score_before[p1], score_before[p2], score_after[p1], score_after[p2], races
    else:
        if WITH_STATE:
            return [], [], [], [], [], [], set(), set(), [], [], {}
        else:
            return [], [], [], [], [], [], {}


def format_battle_for_clust(players_races, armies_battle):
    """ take an "extract_armies_battles" formatted battle data and make it
    ready for clustering (ex: 'Protoss Dragoon': 12, 'Protoss Zealot': 6)
    returning 1 dict of vectors of units numbers per unit types, per race """
    r = {'P': [], 'T': [], 'Z': []}
    for k,v in players_races.iteritems():
        if k in armies_battle[1]: # armies_battle[1] is "max units engaged"
            r[v].append(copy.deepcopy(armies_battle[1][k]))
    return r


def scale_up_special(l, race):
    """ used with SCALE_UP_SPECIAL_UNITS on percent lists (with are then 
    no longer percents...) to give more importance to special units """
    tmp = l
    for i, ut in enumerate(ArmyCompositions.ut_by[race]):
        if ut in ArmyCompositions.special[race] and tmp[i] > 0.0001:
            tmp[i] += ArmyCompositions.special[race][ut]
    return tmp
            

class ArmyCompositions:
    ut_by_race = {}
    for race in ['T', 'P', 'Z']:
        ut_by_race[race] = unit_types.by_race.military[race]+[unit_types.by_race.drop[race]]
        if WITH_STATIC_DEFENSE:
            ut_by_race[race].extend(unit_types.by_race.static_defense[race])
        if WITH_WORKERS:
            ut_by_race[race].append(unit_types.by_race.workers[race])
    ut_by = copy.deepcopy(ut_by_race)
    ut_by['T'].pop(ut_by['T'].index('Terran Siege Tank Siege Mode'))
    special = { # list special units by race and their multipliers in the list of percents
            'P': {'Protoss Observer': 10000,
                'Protoss Arbiter': 0, #10000,
                'Protoss High Templar': 0,
                'Protoss Dark Archon': 0,
                'Protoss Shuttle': 0},
            'T': {'Terran Science Vessel': 10000,
                'Terran Siege Tank Tank Mode': 0,
                'Terran Ghost': 0,
                'Terran Dropship': 0},
            'Z': {'Zerg Queen': 0,
                'Zerg Lurker': 0,
                'Zerg Defiler': 0, #10000
                'Zerg Overlord': 10000}#100000}
            }

    ac_by_race = {}

    def __init__(self, race):
        self.race = race
        if race == 'P':
            self.basic_units = {
                    'P_zealot': {'Protoss Zealot': 0.4},
                    'P_goon': {'Protoss Dragoon': 0.3},
                    'P_DT': {'Protoss Dark Templar': 0.6},
                    'P_reaver': {'Protoss Reaver': 0.1},
                    'P_carrier': {'Protoss Carrier': 0.2},
                    'P_corsair': {'Protoss Corsair': 0.2},
                    'P_scout': {'Protoss Scout': 0.4},
                    'P_archon': {'Protoss Archon': 0.2}}
            self.special_units = {
                    'P_observer': {'Protoss Observer': 0.01},
                    'P_arbiter': {'Protoss Arbiter': 0.05},
                    'P_HT': {'Protoss High Templar': 0.05},
                    'P_darchon': {'Protoss Dark Archon': 0.05},
                    'P_shuttle': {'Protoss Shuttle': 0.2} # full with zealots
                    }
        elif race == 'T':
            self.basic_units = {
                    'T_marine': {'Terran Marine': 0.5},
                    'T_medic': {'Terran Medic': 0.1},
                    'T_firebat': {'Terran Firebat': 0.1},
                    'T_vulture': {'Terran Vulture': 0.3, 'Terran Vulture Spider Mine': 0.05},
                    'T_goliath': {'Terran Goliath': 0.2},
                    'T_wraith': {'Terran Wraith': 0.5},
                    'T_battlecruiser': {'Terran Battlecruiser': 0.2},
                    'T_valkyrie': {'Terran Valkyrie': 0.2}}
            self.special_units = {
                    'T_vessel': {'Terran Science Vessel': 0.05},
                    'T_tank': {'Terran Siege Tank Tank Mode': 0.2}, # trick
                    'T_ghost': {'Terran Ghost': 0.1},
                    'T_dropship': {'Terran Dropship': 0.1} # full with marines
                    }
        elif race == 'Z':
            self.basic_units = {
                    'Z_zergling': {'Zerg Zergling': 0.7},
                    'Z_hydra': {'Zerg Hydralisk': 0.4},
                    'Z_ultra': {'Zerg Ultralisk': 0.15},
                    'Z_muta': {'Zerg Mutalisk': 0.5},
                    'Z_guardian': {'Zerg Guardian': 0.2},
                    'Z_devourer': {'Zerg Devourer': 0.2},
                    'Z_scourge': {'Zerg Scourge': 0.2}}
            self.special_units = {                    
                    'Z_queen': {'Zerg Queen': 0.05},
                    'Z_lurker': {'Zerg Lurker': 0.2},
                    'Z_defiler': {'Zerg Defiler': 0.05},
                    'Z_overlord': {'Zerg Overlord': 0.05} # full with zerglings or detector
                    }
        self.compositions = {}
        self.compositions.update(self.basic_units)
        self.compositions.update(self.special_units)
        for unit,value in self.basic_units.iteritems():
            for sunit,svalue in self.special_units.iteritems():
                self.compositions[unit+'_'+sunit[2:]] = {}
                self.compositions[unit+'_'+sunit[2:]].update(value)
                self.compositions[unit+'_'+sunit[2:]].update(svalue)
        self.compo_count = {}
        self.P_unit_knowing_cluster = {}
        def f(min_p, max_p, x):
            if min_p < x < max_p:
                return disc_width*1.0/(max_p-min_p) - (1.0-max_p+min_p)*epsilon
            else:
                return epsilon
        for name,compo in self.compositions.iteritems():
            for unit,min_percentage in compo.iteritems():
                max_percentage = 1.0-sum([v for k,v in compo.iteritems() if k != unit])
                if unit not in self.P_unit_knowing_cluster:
                    self.P_unit_knowing_cluster[unit] = defaultdict(lambda: lambda x: disc_width)
                
                self.P_unit_knowing_cluster[unit].update({name: functools.partial(f, min_percentage, max_percentage)})
        self.n_units = len(self.P_unit_knowing_cluster)
        self.register()


    def register(self):
        ArmyCompositions.ac_by_race[self.race] = self


    def d_prod_Ui_C(self, percents_list):
        """ Computes ∏_i P(U_i|C=c) ∀ clusters c in C, returns {c: logprob} """
        d = {}
        for cluster in self.compositions:
            d[cluster] = 0.0
            for i, unit_type in enumerate(ArmyCompositions.ut_by[self.race]):
                d[cluster] += math.log(self.P_unit_knowing_cluster[unit_type][cluster](percents_list[i]))
        return d


    def prod_Ui_C(self, percents_list):
        """ Computes ∏_i P(U_i|C=c) ∀ clusters c in C, returns {c: prob} """
        d = {}
        for cluster in self.compositions:
            d[cluster] = 1.0
            for i, unit_type in enumerate(ArmyCompositions.ut_by[self.race]):
                d[cluster] *= self.P_unit_knowing_cluster[unit_type][cluster](percents_list[i])
        return np.array(d.values())


    def tabulate(self, disc_steps):
        tmp = np.ndarray(shape=(len(disc_steps),
            self.n_units,
            len(self.compositions)), dtype='float')
        for unit in self.P_unit_knowing_cluster:
            for cluster, prob in self.P_unit_knowing_cluster[unit].iteritems():
                for i, val in enumerate(disc_steps):
                    tmp[i][ArmyCompositionModel.unit_to_int(unit)][ArmyCompositionModel.cluster_to_int(cluster)] = prob(val)
        return tmp


    def count(self, percents_list):
        """ Adds log probabilities of clusters for a given battle """
        dist = ArmyCompositions.ac_by_race[race].d_prod_Ui_C(p_l) 
        for c, logprob in dist.iteritems():
            self.compo_count[c] = self.compo_count.get(c, 0) + logprob
    

    def prune(self, tail_numbers_to_prune=0, numbers_to_keep=0):
        """ 
        Removes really unprobable cluster (which were never seen) 
        Additionally, can remove tail_numbers_to_prune clusters from the
        tail of the clusters probability distribution (less probable ones),
        or keep only numbers_to_keep top (most probable) clusters
        """
        mini = 0
        to_rem = set()
        # insert the clusters to remove in to_rem list
        for clust, sumlogprob in self.compo_count.iteritems():
            if sumlogprob < mini:
                mini = sumlogprob
                to_rem = set()
                to_rem.add(clust)
            elif sumlogprob < mini+1:
                to_rem.add(clust)
        # TODO TEST:
        if tail_numbers_to_prune > 0 or numbers_to_keep > 0:
            s = sorted([(c,p) for c,p in self.compo_count.iteritems()], key=lambda x: x[1])
            s = [x[0] for x in s]
            if tail_numbers_to_prune > 0:
                if tail_numbers_to_prune > len(s):
                    print >> sys.stderr, "ERROR: must prune more clusters than existing"
                    sys.exit(-1)
                to_rem.update(set(s[:tail_numbers_to_prune]))
            elif numbers_to_keep > 0:
                s.reverse()
                to_rem.update(set(s[numbers_to_keep:]))
        # /TODO TEST
        # do the actual removing
        for clust in to_rem:
            self.compositions.pop(clust)
            for unit in self.P_unit_knowing_cluster:
                self.P_unit_knowing_cluster[unit].pop(clust, 0)
        if DEBUG_OUR_CLUST:
            print "Clusters:"
            for clust in self.compositions:
                print clust


class ArmyCompositionsPCA(ArmyCompositions):
    def __init__(self, race, n_components=5):
        from sklearn import decomposition
        self.pca = decomposition.PCA(n_components=n_components)
        self.compositions = range(n_components)
        self.n_units = len(ArmyCompositions.ut_by[race])
        self.race = race
        self.data = []
        self.register()


    def d_prod_Ui_C(self, percents_list):
        lpc = self.pca.transform(np.array(percents_list))
        return dict(zip(self.compositions, lpc))


    def prod_Ui_C(self, percents_list):
        return self.pca.transform(np.array(percents_list))


    def count(self, p_l):
        self.data.append(p_l)


    def prune(self):
        """ Fit the PCA """
        self.data = np.array(self.data)
        self.pca.fit(self.data)


class ArmyCompositionsICA(ArmyCompositions):
    def __init__(self, race, n_components=6):
        from sklearn import decomposition
        self.ica = decomposition.FastICA(n_components=n_components)
        self.compositions = range(n_components)
        self.n_units = len(ArmyCompositions.ut_by[race])
        self.race = race
        self.data = []
        self.register()


    def d_prod_Ui_C(self, percents_list):
        lpc = self.ica.transform(np.array(percents_list))
        return dict(zip(self.compositions, lpc))


    def prod_Ui_C(self, percents_list):
        return self.ica.transform(np.array(percents_list))


    def count(self, p_l):
        self.data.append(p_l)


    def prune(self):
        """ Fit the PCA """
        self.data = np.array(self.data)
        self.ica.fit(self.data)


class ArmyCompositionsIsomap(ArmyCompositions):
    def __init__(self, race, n_components=6):
        from sklearn import manifold
        self.iso = manifold.Isomap(n_components=n_components)
        self.compositions = range(n_components)
        self.n_units = len(ArmyCompositions.ut_by[race])
        self.race = race
        self.data = []
        self.register()


    def d_prod_Ui_C(self, percents_list):
        tmp = self.iso.transform(np.array([percents_list]))
        return dict(zip(self.compositions, tmp[0]))


    def prod_Ui_C(self, percents_list):
        return self.iso.transform(np.array([percents_list]))[0]


    def count(self, p_l):
        self.data.append(p_l)


    def prune(self):
        """ Fit the PCA """
        self.data = np.array(self.data)
        self.iso.fit(self.data)


class ArmyCompositionsKmeans(ArmyCompositions):
    def __init__(self, race, n_components):
        from sklearn import cluster
        self.km = cluster.KMeans(k=n_components, n_jobs=4)
        self.compositions = range(n_components)
        self.n_units = len(ArmyCompositions.ut_by[race])
        self.race = race
        self.data = []
        self.register()


    def d_prod_Ui_C(self, percents_list):
        return dict(zip(self.compositions, self.prod_Ui_C(percents_list)))


    def prod_Ui_C(self, percents_list):
        tmp = np.array([0.0 for i in self.compositions])
        tmp[self.km.predict(np.array([percents_list]))] = 1.0
        return np.array(tmp)


    def count(self, p_l):
        self.data.append(p_l)


    def prune(self):
        """ Fit the KMeans. """
        self.data = np.array(self.data)
        self.km.fit(self.data)


import gensim
PERCENTS_TO_INT = False
class ArmyCompositionsLDA(ArmyCompositions):
    def __init__(self, race, n_components):
        self.lda = None
        self.compositions = range(n_components)
        self.n_units = len(ArmyCompositions.ut_by[race])
        self.race = race
        self.data = []
        self.register()


    def d_prod_Ui_C(self, percents_list):
        if PERCENTS_TO_INT:
            percents_list = [int(x*1000) for x in percents_list]
        tmp = self.lda[zip(range(self.n_units), percents_list)]
        return dict(tmp)


    def prod_Ui_C(self, percents_list):
        if PERCENTS_TO_INT:
            percents_list = [int(x*1000) for x in percents_list]
        tmp = self.lda[zip(range(self.n_units), percents_list)]
        return [t[1] for t in tmp]


    def count(self, p_l):
        if PERCENTS_TO_INT:
            self.data.append([int(x*1000) for x in p_l])
        else:
            self.data.append(p_l)


    def prune(self):
        """ Fit the LDA """
        self.data = [zip(range(self.n_units), line) for line in self.data]
        #print len(self.data[0])
        #print self.n_units
        #print ArmyCompositions.ut_by[self.race]
        #print self.data
        id2word = dict(zip(range(self.n_units), ArmyCompositions.ut_by[self.race]))
        self.lda = gensim.models.ldamodel.LdaModel(corpus=self.data, id2word=id2word, num_topics=len(self.compositions), update_every=0, passes=100)
        print self.lda
        for compo in self.compositions:
            print self.lda.print_topic(compo)
        #print self.lda.print_topics(len(self.compositions))



class ArmyCompositionsGMM(ArmyCompositions):
    def __init__(self, race, n_components=0):
        from sklearn import mixture
        if n_components != 0:
            self.gmm = mixture.GMM(n_components=n_components, covariance_type='full')
        else:
            if SERIALIZE_GMM:
                self.gmm = [mixture.GMM(n_components=i, covariance_type='full') for i in range(3,13)]
            else:
                self.gmm = [mixture.GMM(n_components=i, covariance_type=cv) for i in range(3,11) for cv in ['spherical', 'tied', 'diag', 'full']]
        self.compositions = range(n_components)
        self.n_units = len(ArmyCompositions.ut_by[race])
        self.race = race
        self.data = []
        self.register()


    def d_prod_Ui_C(self, percents_list):
        tmp = self.gmm.predict_proba(np.array([percents_list]))
        return dict(zip(self.compositions, tmp[0]))


    def prod_Ui_C(self, percents_list):
        return self.gmm.predict_proba(np.array([percents_list]))[0]


    def count(self, p_l):
        self.data.append(p_l)


    def prune(self):
        """ Fit the GMM. If there is no specified number of clusters 
        (components), select the lowest BIC GMM """
        self.data = np.array(self.data)
        if self.compositions == []:
            best_gmm = 0
            best_bic = 1e30
            for i, g in enumerate(self.gmm):
                g.fit(self.data)
                tmp = g.bic(self.data)
                if tmp < best_bic:
                    best_gmm = i
                    best_bic = tmp
            self.gmm = self.gmm[best_gmm]
        else:
            self.gmm.fit(self.data)

        if SERIALIZE_GMM:
            wf = open(self.race + '.gmm', 'w')
            tmpstr = "n compo: "
            tmpstr += str(self.gmm.n_components) + '\n'
            tmpstr += "n features: "
            tmpstr += str(len(self.gmm.means_[0])) + '\n'
            tmpstr += "in order:\n" 
            tmpstr += ";".join(ArmyCompositions.ut_by[self.race]) + '\n'
            tmpstr += "means:\n"
            for comp in range(len(self.gmm.means_)):
                tmpstr += "component" + str(comp) + '\n'
                tmpstr += ";".join(map(str, self.gmm.means_[comp])) + '\n'
            tmpstr += "covars:\n"
            for comp in range(len(self.gmm.covars_)):
                tmpstr += "component" + str(comp) + '\n'
                for feat in range(len(self.gmm.covars_[0])):
                    tmpstr += "feature" + str(feat) + '\n'
                    tmpstr += ";".join(map(str, self.gmm.covars_[comp][feat])) + '\n'
            tmpstr += "weights:\n"
            tmpstr += ";".join(map(str, self.gmm.weights_))
            wf.write(tmpstr)

        self.compositions = range(self.gmm.n_components)
        if DEBUG_GMM:
            print >> sys.stderr, "n components:", len(self.gmm.means_), "cv:", self.gmm.covars_
            #print >> sys.stderr, self.gmm.predict(self.data)
            from sklearn import decomposition, lda, manifold
            X_pca = decomposition.RandomizedPCA(n_components=2).fit_transform(self.data)
            y = self.gmm.predict(self.data)
            n = len(self.data)
            plot_embedding(n, y, X_pca, 
                    'GMM_PCA_'+self.race, "PCA projection of the clusters")
            #data2 = self.data.copy()
            #data2.flat[::self.data.shape[1] + 1] += 0.01
            #X_lda = lda.LDA(n_components=2).fit_transform(data2, y)
            #plot_embedding(n, y, X_lda, 
            #        'GMM_LDA_'+self.race, "LDA projection of the clusters")
            X_iso = manifold.Isomap(n_components=2).fit_transform(self.data)
            plot_embedding(n, y, X_iso, 
                    'GMM_ISO_'+self.race, "Isomap projection of the clusters")
            X_lle = manifold.LocallyLinearEmbedding(n_components=2, method='standard').fit_transform(self.data)
            plot_embedding(n, y, X_lle, 
                    'GMM_LLE_'+self.race, "Locally Linear Embedding of the clusters")



def plot_embedding(n, y, X, save_name, title=None):
    x_min, x_max = np.min(X, 0), np.max(X, 0)
    X = (X - x_min) / (x_max - x_min)
    pl.figure()
    ax = pl.subplot(111)
    for i in range(n):
        pl.text(X[i, 0], X[i, 1], str(y[i]),
                color=pl.cm.Set1(y[i] / 10.),
                fontdict={'weight': 'bold', 'size': 11})
    pl.xticks([]), pl.yticks([])
    if title is not None:
        pl.title(title)
    pl.savefig(save_name+'.png')
                

class ArmyCompositionsDPGMM(ArmyCompositions):
    def __init__(self, race, n_components=0):
        from sklearn import mixture
        if n_components == 0:
            self.dpgmm = mixture.DPGMM(n_components=42, covariance_type='full')
        else:
            self.dpgmm = mixture.DPGMM(n_components=n_components, covariance_type='full')
        self.compositions = range(n_components)
        self.n_units = len(ArmyCompositions.ut_by[race])
        self.race = race
        self.data = []
        self.register()


    def d_prod_Ui_C(self, percents_list):
        tmp = self.dpgmm.predict_proba(np.array([percents_list]))
        return dict(zip(self.compositions, tmp[0]))


    def prod_Ui_C(self, percents_list):
        return self.dpgmm.predict_proba(np.array([percents_list]))[0]


    def count(self, p_l):
        self.data.append(p_l)


    def prune(self):
        """ Fit the DPGMM """ 
        self.data = np.array(self.data)
        self.dpgmm.fit(self.data)
        if self.compositions == []:
            nc = 42
            Y = self.dpgmm.predict(self.data)
            for i in range(len(self.dpgmm.means_)):
                if not np.any(Y == i):
                    nc -= 1
            self.compositions = range(nc)
        

class ArmyCompositionModel:
    @staticmethod
    @memoize
    def unit_to_int(unit):
        if unit == 'Terran Siege Tank Siege Mode':
            unit = 'Terran Siege Tank Tank Mode'
        return ArmyCompositions.ut_by[unit[0]].index(unit)

    @staticmethod
    @memoize
    def cluster_to_int(cluster):
        return ArmyCompositions.ac_by_race[cluster[0]].compositions.keys().index(cluster)

    @staticmethod
    @memoize
    def int_to_cluster(i, r):
        return ArmyCompositions.ac_by_race[r].compositions.keys()[i]

    @staticmethod
    @memoize
    def c_possible_under_tt(c, enum, tt):
        """ c is the cluster, tt is the techtree in vector_X set format,
        enum is the mapping between names to vector_X indices """
        for ut in ArmyCompositions.ac_by_race[c[0]].compositions[c]:
            for req in unit_types.required_for(ut):
                if enum.index(req) not in tt:
                    return False
        return True

    def Cfinal_knowing_CtacticsCcounter(self, tts, P_Ctactics, P_Ccounter):
        # Which values of C are compatible with tts? (set of buildings we have)
        def has_all_requirements(list_ut):
            for ut in list_ut:
                for req in unit_types.required_for(ut):
                    if req not in tts:
                        return False
            return True
        tmp = []
        for i in range(self.W_knowing_Ccounter_ECnext[1].shape[0]):
            cluster = ArmyCompositionModel.int_to_cluster(i, self.race)
            if has_all_requirements(ArmyCompositions.ac_by_race[self.race].compositions[cluster].keys()):
                tmp.append(1.0)
            else:
                tmp.append(0.0)
        d1 = np.array(tmp)
        # Fusion values for Ctactics and Ccounter
        d2 = self.alpha*P_Ctactics + (1-self.alpha)*P_Ccounter
        return d1*d2 # TODO verify
    
    def __init__(self, mu, ac, eac, alpha=0.25):
        """
        Takes two ArmyComposition objects (for us and for the enemy) and
        and builds an ArmyCompositionModel
        """
        assert (mu[0] == ac.race and mu[2] == eac.race)
        self.n_train = 0
        self.alpha = alpha
        self.race = ac.race
        self.erace = eac.race
        self.matchup = self.race + 'v' + self.erace
        self.disc_steps = np.arange(0, 1, disc_width)
        self.tech_trees = vector_X(mu[0], mu[2]) # gives techtrees of mu[2]

        # P(Cfinal)
        self.Cfinal = np.ones(len(ac.compositions)) # can put a prior here...
        self.Cfinal /= sum(self.Cfinal)

        # P(EC)
        self.EC = np.ones(len(eac.compositions)) # ...or learn it!
        self.EC /= sum(self.EC)

        # P(EC^t|EC^{t+1}) learned transitions
        self.EC_knowing_ECnext = np.ndarray(shape=(len(eac.compositions),
            len(eac.compositions)), dtype='float')
        self.EC_knowing_ECnext.fill(ADD_SMOOTH_EC_EC)

        # P(EC^{t+1}|ETT) possible EC under ETT _OR_ learned correlations
        if LEARNED_EC_KNOWING_ETT:
            self.ECnext_knowing_ETT = np.ndarray(shape=(len(eac.compositions),
                len(self.tech_trees.vector_X)), dtype='float')
            self.ECnext_knowing_ETT.fill(ADD_SMOOTH_EC_TT)
        else:
            pass
        # else -> function 1.0 for ec compatibles with ett, else 0.0

        # P(Win|C_counter,EC^{t+1}) learned correlations
        self.W_knowing_Ccounter_ECnext = np.ndarray(shape=(2, # 0,1: Lose,Win
            len(ac.compositions),
            len(eac.compositions)), dtype='float')
        self.W_knowing_Ccounter_ECnext.fill(ADD_SMOOTH_C_EC)
        
        # P(U|C_final) and P(EU|EC) from ac and eac
        self.prod_Ui_Cfinal = ac.prod_Ui_C
        self.prod_EU_EC = eac.prod_Ui_C


    def train_W_knowing_C_EC(self, battle, with_efficiency=False):
        def efficiency(us_before, us_after, them_before, them_after):
            """ compute the efficiency of the battle our ("us") POV """
            our_loss = us_before-us_after
            their_loss = them_before-them_after
            our_advantage = us_before/(them_before+0.0001)
            our_efficiency = their_loss/(our_loss+0.0001)
            return min(2.0, our_efficiency/our_advantage)

        # TODO TODO TODO TODO TODO Fix + Clean

        w, l = get_winner_loser(battle)
        w_a, l_a = battle[2+w], battle[2+l] # init (total) army scores
        w_s, l_s = battle[4+w], battle[4+l] # final scores
        races = battle[-1]
        w_p = percent_list.dict_to_list(battle[w], battle[-1][w])
        l_p = percent_list.dict_to_list(battle[l], battle[-1][l])

#        print "================================"
#        print battle
#        win_total = 0.0
#        lose_total = 0.0

        winner_efficiency = efficiency(w_a, w_s, l_a, l_s)
        if races[w] == self.race:
            distrib_C_us = self.prod_Ui_Cfinal(w_p)
            distrib_C_them = self.prod_EU_EC(l_p)
            for c, p in enumerate(distrib_C_us):
                for ec, ep in enumerate(distrib_C_them):
                    if not with_efficiency:
                        self.W_knowing_Ccounter_ECnext[1,c,ec] += p*ep
                    else:
                        self.W_knowing_Ccounter_ECnext[1,c,ec] += p*ep*winner_efficiency
#                    win_total += p*ep*winner_efficiency
#            print "winner, added:", win_total

        if races[l] == self.race:
            distrib_C_us = self.prod_Ui_Cfinal(l_p)
            distrib_C_them = self.prod_EU_EC(w_p)
            for c, p in enumerate(distrib_C_us):
                for ec, ep in enumerate(distrib_C_them):
                    if not with_efficiency:
                        self.W_knowing_Ccounter_ECnext[0,c,ec] += p*ep
                    else:
                        self.W_knowing_Ccounter_ECnext[0,c,ec] += p*ep*winner_efficiency
#                    lose_total += p*ep*winner_efficiency
#            print "loser, added:", lose_total
#        print "================================"


    def train_Ctplus1_knowing_TT(self, battle):
        def learn_for_p(player):
            distrib_C = self.prod_EU_EC(percent_list.dict_to_list(battle[player], battle[-1][player]))
            tt_to_count = []
            for i, tt in enumerate(self.tech_trees.vector_X):
                if len(tt) >= len(battle[6+player]):
                    good = True
                    for building in battle[6+player]: # battle[6+player] = 
                                                      # tech_tree[player]
                        if self.tech_trees.index_enum(building) not in tt:
                            good = False
                            break
                    if good:
                        tt_to_count.append(i)
            for c, p in enumerate(distrib_C):
                for i in tt_to_count:
                    self.ECnext_knowing_ETT[c,i] += p

        p1r, p2r = battle[-1][0], battle[-1][1]
        if p1r == self.erace:
            learn_for_p(0)
        if p2r == self.erace:
            learn_for_p(1)


    def train_ECtplus1_knowing_ECt(self, battle):
        def learn_for_p(player):
            distrib_ECtplus1 = self.prod_EU_EC(percent_list.dict_to_list(battle[player], battle[-1][player]))
            distrib_ECt = self.prod_EU_EC(percent_list.dict_to_list(battle[8+player], battle[-1][player]))
            for ect, p in enumerate(distrib_ECt):
                for ectplus1, pp in enumerate(distrib_ECtplus1):
                    self.EC_knowing_ECnext[ect,ectplus1] += p*pp

        p1r, p2r = battle[-1][0], battle[-1][1]
        if p1r == self.erace:
            learn_for_p(0)
        if p2r == self.erace:
            learn_for_p(1)


    def train(self, battle, with_efficiency=False):
        self.n_train += 1
        ### *********** train P(W|C,EC) ***********
        self.train_W_knowing_C_EC(battle, with_efficiency)
        ### *********** train P(C^{t+1}|TT) ***********
        if LEARNED_EC_KNOWING_ETT:
            self.train_Ctplus1_knowing_TT(battle)
        ### *********** train P(EC^t|EC^{t+1}) ***********
        self.train_ECtplus1_knowing_ECt(battle)

        
    def normalize(self):
        for ecn in range(self.EC_knowing_ECnext.shape[1]):
            self.EC_knowing_ECnext[:,ecn] /= sum(self.EC_knowing_ECnext[:,ecn])

        if LEARNED_EC_KNOWING_ETT:
            for ett in range(self.ECnext_knowing_ETT.shape[1]):
                self.ECnext_knowing_ETT[:,ett] /= sum(self.ECnext_knowing_ETT[:,ett])

        for cn in range(self.W_knowing_Ccounter_ECnext.shape[1]):
            for ecn in range(self.W_knowing_Ccounter_ECnext.shape[2]):
                self.W_knowing_Ccounter_ECnext[:,cn,ecn] /= sum(self.W_knowing_Ccounter_ECnext[:,cn,ecn])

        for cn in range(self.W_knowing_Ccounter_ECnext.shape[1]):
            for ecn in range(self.W_knowing_Ccounter_ECnext.shape[2]):
                self.W_knowing_Ccounter_ECnext[:,cn,ecn] /= sum(self.W_knowing_Ccounter_ECnext[:,cn,ecn])

        if PLOT_EC_KNOWING_ECNEXT:
            fig = pl.figure()
            fig.suptitle('P(EC^{t}|EC^{t+1})')
            ax = fig.add_subplot(111)
            im = pl.pcolor(self.EC_knowing_ECnext)
            pl.colorbar(im)
            ax.images.append(im)
            ax.set_ylabel("EC^{t}")
            ax.set_xlabel("EC^{t+1}")
            pl.savefig(self.erace+"_EC_knowing_ECnext.png")

        if PLOT_W_KNOWING_C_EC:
            fig = pl.figure()
            fig.suptitle('P(win|C,EC)')
            ax = fig.add_subplot(111)
            im = pl.pcolor(self.W_knowing_Ccounter_ECnext[1,:,:])
            pl.colorbar(im)
            ax.images.append(im)
            ax.set_ylabel("C")
            ax.set_xlabel("EC")
            pl.savefig(self.erace+"_W_knowing_C_EC.png")
            
        if PLOT_ECNEXT_KNOWING_ETT:
            ### TODO review this plotting
#            from matplotlib.image import NonUniformImage
#            fig = pl.figure()
#            fig.suptitle('P(EC^{t+1}|ETT^{t})')
#            ax = fig.add_subplot(111)
#            K = len(self.ECnext_knowing_ETT)
#            V = len(self.ECnext_knowing_ETT[0])
#            print self.ECnext_knowing_ETT[0]
#            print self.ECnext_knowing_ETT[1]
#            print self.ECnext_knowing_ETT[2]
#            im = NonUniformImage(ax, interpolation='nearest', extent=(-0.5,V-0.5,-0.5,K-0.5))
#            x = np.linspace(0, K-1, K)
#            y = np.linspace(0, V-1, V)
#            #z = np.array([self.EC_knowing_ECnext[int(i),int(j)] for i in x for j in y])
#            im.set_data(y, x, np.log(self.ECnext_knowing_ETT))
#            ax.images.append(im)
#            ax.set_xlim(-0.5,V-0.5)
#            ax.set_ylim(-0.5,K-0.5)
#            ax.set_ylabel("EC^{t+1}")
#            ax.set_xlabel("ETT^{t}")
#            pl.savefig(self.erace+"_ECnext_knowing_ETT.png")
            #for i in range(self.ECnext_knowing_ETT.shape[0]):
            width = 0.5
            for i in range(self.ECnext_knowing_ETT.shape[1]):
                fig = pl.figure()
                #fig.suptitle('P(EC^{t+1}='+i+'|ETT)')
                fig.suptitle('P(EC^{t+1}|ETT^{t}='+str(i)+')')
                #s = round(math.sqrt(self.ECnext_knowing_ETT.shape[1])+0.5)
                ax = fig.add_subplot(111)
                ind = np.arange(len(self.ECnext_knowing_ETT))
                #ax.set_ylabel('P(EC^{t+1}|ETT)')
                ax.set_ylabel('P(EC)')
                ax.set_xlabel('EC')
                ax.bar(ind, np.log(1+self.ECnext_knowing_ETT[:,i]), width, color='r')
                #ax.bar(ind, self.ECnext_knowing_ETT[:,i], width, color='r')
                fig.savefig('plots_tt/'+self.erace+'_ECnext_knowing_ETT_'+str(i)+'.png')

        if SHOW_NORMALIZE_OUTPUT:
            print "U knowing C_final", self.U_knowing_Cfinal
            print "==================================="
            print "EU knowing EC", self.EU_knowing_EC
            print "==================================="
            print "W=true knowing C,EC"
            for c in range(self.W_knowing_Ccounter_ECnext.shape[1]):
                print "vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv"
                print ArmyCompositions.ac_by_race[self.race].compositions.keys()[c]
                print ""
                print filter(lambda x: abs(x[1]-0.5) > 0.01, zip(ArmyCompositions.ac_by_race[self.erace].compositions.keys(), self.W_knowing_Ccounter_ECnext[1][c]))
                print "" 
                print filter(lambda x: abs(x[1]-0.5) > 0.01, zip(ArmyCompositions.ac_by_race[self.erace].compositions.keys(), self.W_knowing_Ccounter_ECnext[0][c]))
                print "^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^"
            print "==================================="

    def winner_battle_C_EC_only(self, battle, most_probable=False):
        """ use P(W|C,EC).P(C).P(EC) to determine the winner """
        p1_p = percent_list.dict_to_list(battle[0], battle[-1][0])
        distrib_C_p1 = self.prod_Ui_Cfinal(p1_p)
        p2_p = percent_list.dict_to_list(battle[1], battle[-1][1])
        distrib_C_p2 = self.prod_EU_EC(p2_p)
        t = 0.0
        pep = 0.0
        for c, p in enumerate(distrib_C_p1):
            for ec, ep in enumerate(distrib_C_p2):
                if most_probable:
                    if p*ep > pep:
                        t = self.W_knowing_Ccounter_ECnext[1,c,ec]
                        pep = p*ep
                else:
                    t += self.W_knowing_Ccounter_ECnext[1,c,ec] * p * ep
        # if t > 0.5 it means C beats EC, otherwise C loses against EC
        if t >= 0.5:
            return 0
        else:
            return 1

    def winner_battle(self, battle, most_probable=False):
        """ use P(W|C,EC).P(C).P(EC) + score before to determine the winner """
        p1_p = percent_list.dict_to_list(battle[0], battle[-1][0])
        distrib_C_p1 = self.prod_Ui_Cfinal(p1_p)
        p2_p = percent_list.dict_to_list(battle[1], battle[-1][1])
        distrib_C_p2 = self.prod_EU_EC(p2_p)

        t = 0.0
        pep = 0.0
        for c, p in enumerate(distrib_C_p1):
            for ec, ep in enumerate(distrib_C_p2):
                if most_probable:
                    if p*ep > pep:
                        t = self.W_knowing_Ccounter_ECnext[1,c,ec]
                        pep = p*ep
                else:
                    t += self.W_knowing_Ccounter_ECnext[1,c,ec] * p * ep
        #for c, p in enumerate(distrib_C_p1):
        #    for ec, ep in enumerate(distrib_C_p2):
        #        t -= self.W_knowing_Ccounter_ECnext[0,c,ec] * p * ep

        # if t > 0.5 it means C beats EC, otherwise C loses against EC
        factor = t*2
#        print ">>> winner battle:"
#        print battle[0]
#        print filter(lambda x: x[1] > 0.001, zip(ArmyCompositions.ac_by_race[battle[-1][0]].compositions, distrib_C_p1))
#        print battle[1]
#        print filter(lambda x: x[1] > 0.001, zip(ArmyCompositions.ac_by_race[battle[-1][1]].compositions, distrib_C_p2))
#        print "factor:", factor
        if battle[2]*factor > battle[3]:
            return 0
        else:
            return 1
        


class percent_list(list):
    """ a type of list which adds percentages of units types in units order """
    @staticmethod
    def dict_to_list(d, race=None):
        if race == None:
            race = d.iterkeys().next()[0] # first character of the first unit
        tmp = [d.get(u, 0.0) for u in ArmyCompositions.ut_by_race[race]]
        if race == 'T':
            tmp[ArmyCompositions.ut_by_race[race].index('Terran Siege Tank Tank Mode')] += tmp.pop(ArmyCompositions.ut_by_race[race].index('Terran Siege Tank Siege Mode'))
        if SCALE_UP_SPECIAL_UNITS:
            tmp = scale_up_special(tmp, race)
        return tmp

    def new_battle(self, d, race=None):
        self.append(percent_list.dict_to_list(d, race))


def matchup(race, d):
    """ determines the match-up beginning by race with the 
    armies_battles_for_clust dictionary 'd' """
    mu = race + 'v'
    for k,v in d.iteritems():
        if len(v) > 0 and k != race:
            return mu+k
    for k,v in d.iteritems():
        if len(v) > 0:
            return mu+k


def get_winner_loser(b):
    # b[-5] = score player 1, b[-4] = score player 2
    if b[-4] > b[-5]:
        return 1, 0
    else:
        return 0, 1


f = sys.stdin
armies_battles_for_regr = []
armies_battles_for_clust = {'P': percent_list(), 
        'T': percent_list(), 
        'Z': percent_list()}
battles_for_clustering = []
fnamelist = []

#ArmyCompositions('P')
#ArmyCompositions('T')
#ArmyCompositions('Z')

#ArmyCompositionsPCA('P')
#ArmyCompositionsPCA('T')
#ArmyCompositionsPCA('Z')

#ArmyCompositionsICA('P')
#ArmyCompositionsICA('T')
#ArmyCompositionsICA('Z')

#ArmyCompositionsIsomap('P')
#ArmyCompositionsIsomap('T')
#ArmyCompositionsIsomap('Z')

#ArmyCompositionsKmeans('P', 6)
#ArmyCompositionsKmeans('T', 6)
#ArmyCompositionsKmeans('Z', 6)

ArmyCompositionsGMM('P')
ArmyCompositionsGMM('T')
ArmyCompositionsGMM('Z')

#ArmyCompositionsDPGMM('P', 6)
#ArmyCompositionsDPGMM('T', 6)
#ArmyCompositionsDPGMM('Z', 6)

#ArmyCompositionsLDA('P', 6)
#ArmyCompositionsLDA('T', 6)
#ArmyCompositionsLDA('Z', 6)

armies_compositions_models = {}

if __name__ == "__main__":
    if len(sys.argv) > 1:
        if sys.argv[1] == '-d': # -d for directory
            import glob
            fnamelist = glob.iglob(sys.argv[2] + '/*.rgd')
        else:
            fnamelist = sys.argv[1:]
        if SCALE_UP_SPECIAL_UNITS:
            print ArmyCompositions.special
        fnamelist = [fna for fna in fnamelist]
        learngames = [fna for fna in fnamelist]
        testgames = []
        if '-w' in sys.argv:
            WITH_WORKERS = True
        if '-s' in sys.argv:
            WITH_STATIC_DEFENSE = True
        if '-u' in sys.argv:
            SCALE_UP_SPECIAL_UNITS = True
        if '-r' in sys.argv:
            WITH_SCORE_RATIO = False
        if '-t' in sys.argv: # -t for tests
            if NUMBER_OF_TEST_GAMES > len(fnamelist):
                print >> sys.stderr, "Number of test games %d > number of games %d" % (NUMBER_OF_TEST_GAMES, len(fnamelist))
                sys.exit(-1)
            i = 0
            random.seed(0) # no randomness, just sampling
            r = random.randint(0, len(fnamelist)-1)
            while i < NUMBER_OF_TEST_GAMES and i < 5000000:
                if fnamelist[r] not in testgames:
                    i += 1
                    testgames.append(fnamelist[r])
                    if NUMBER_OF_TEST_GAMES > len(fnamelist)*2: # TODO TODO REMOVE
                        learngames.remove(fnamelist[r])
                r = random.randint(0,len(fnamelist)-1)

        print "learning from", len(learngames), "games"
        for fname in learngames:
            f = open(fname)
            players_races = data_tools.players_races(f)

            ### Parse battles and extract armies (before, after)
            raw = extract_armies_battles(f)

            if SCORES_REGRESSION:
                ### Format battles for predict/regression (of the outcome)
                battles_r = map(functools.partial(format_battle_for_regr,
                        players_races), raw)
                armies_battles_for_regr.extend(battles_r)

            ### Format battles for clustering (with armies order P>T>Z)
            ### (army_p1, army_p2, score_before_p1, score_before_p2,
            ### score_after_p1, score_after_p2, races)
            battles_c = filter(lambda x: len(x[0]) and len(x[1]),
                    map(functools.partial(format_battle_for_clust_adv,
                    players_races), raw))
            #print battles_c

            ### save these battles for further use
            battles_for_clustering.extend(battles_c)

            ### Sort armies by race and put inside battles_for_clust
            for b in battles_c:
                if WITH_STATE:
                    assert(len(b) == 11) # enforce format
                else:
                    assert(len(b) == 7) # enforce format
                for race in armies_battles_for_clust.iterkeys():
                    for i,prace in b[-1].iteritems():
                        if prace == race:
                            armies_battles_for_clust[race].new_battle(b[i])
            #print armies_battles_for_clust


        ### Do the clustering
        if PARALLEL_COORDINATES_PLOT:
            from common.parallel_coordinates import parallel_coordinates
        annotated_l = []
        for race, l in armies_battles_for_clust.iteritems():
            if len(l) > 0:
                armies_compositions_models[matchup(race, armies_battles_for_clust)] = 0
                x_l = [u for u in ArmyCompositions.ut_by[race]]
                x_l = map(lambda s: ''.join(s.split(' ')[1:]), x_l)
                x_l.append('MostProbableClust')
                if PARALLEL_COORDINATES_PLOT:
                    parallel_coordinates(l, x_labels=x_l).savefig("parallel_"+race+".png")

                for p_l in l:
                    ArmyCompositions.ac_by_race[race].count(p_l)
                ArmyCompositions.ac_by_race[race].prune()

                for p_l in l:
                    dist = ArmyCompositions.ac_by_race[race].d_prod_Ui_C(p_l) 
                    decreasing_probs_clusters = sorted([(c,prob) for c,prob in dist.iteritems()], key=lambda x: x[1], reverse=True)
                    if DEBUG_OUR_CLUST:
                        print zip(x_l, map(lambda x: "%.2f" % x, p_l))
                        print decreasing_probs_clusters[:5]
                    annotated_l.append(p_l + [decreasing_probs_clusters[0][0]])

                if CSV_ARMIES_OUTPUT:
                    csv = open(race+'_armies.csv', 'w')
                    csv.write(','.join(x_l) + '\n')
                    for line in annotated_l:
                        csv.write(','.join(map(lambda e: str(e), line))+'\n')


        ### Learn the model's parameters
        for mu in armies_compositions_models:
            armies_compositions_models[mu] = ArmyCompositionModel(mu, ArmyCompositions.ac_by_race[mu[0]], ArmyCompositions.ac_by_race[mu[2]])
            for battle in battles_for_clustering:
                # (army_p1, army_p2, score_before_p1, score_before_p2,
                # score_after_p1, score_after_p2, players_races)
                armies_compositions_models[mu].train(battle)
            armies_compositions_models[mu].normalize()
            print mu, "trained on", armies_compositions_models[mu].n_train, "battles"



        if '-t' in sys.argv:
            print "testing from", len(testgames), "games"
            test_battles = []
            for fname in testgames:
                f = open(fname)
                players_races = data_tools.players_races(f)
                raw = extract_armies_battles(f)
                battles_c = filter(lambda x: len(x[0]) and len(x[1]),
                        map(functools.partial(format_battle_for_clust_adv,
                        players_races), raw))
                test_battles.extend(battles_c)

            score_simple_outcome_predictor = 0
            cluster_outcome_predictor = 0
            most_prob_cluster_outcome_predictor = 0
            score_cluster_outcome_predictor = 0
            score_most_prob_outcome_predictor = 0
            for battle in test_battles:
                ### simple outcome predictor: bigger army wins
                if (battle[2]-battle[3])*(battle[4]-battle[5]) > 0:
                    score_simple_outcome_predictor += 1

                ### outcome prediction taking clusters into account
                good1 = False
                good2 = False
                good3 = False
                good4 = False
                mu = battle[-1][0] + 'v' + battle[-1][1]
                if armies_compositions_models[mu].winner_battle_C_EC_only(battle) == get_winner_loser(battle)[0]:
                    cluster_outcome_predictor += 1
                    good1 = True
                if armies_compositions_models[mu].winner_battle_C_EC_only(battle, True) == get_winner_loser(battle)[0]:
                    most_prob_cluster_outcome_predictor += 1
                    good2 = True
                if armies_compositions_models[mu].winner_battle(battle) == get_winner_loser(battle)[0]:
                    score_cluster_outcome_predictor += 1
                    good3 = True
                if armies_compositions_models[mu].winner_battle(battle, True) == get_winner_loser(battle)[0]:
                    score_most_prob_outcome_predictor += 1
                    good4 = True

#                if not good1 or not good2 or not good3 or not good4:
#                mu2 = battle[-1][1] + 'v' + battle[-1][0]
#                print mu
#                print mu2
#                print armies_compositions_models
#                races = {}
#                races[0] = battle[10][1]
#                races[1] = battle[10][0]
#                b = (battle[1], battle[0], battle[3], battle[2], battle[5], battle[4], battle[7], battle[6], battle[9], battle[8], races)
#                if (not good1) and (armies_compositions_models[mu2].winner_battle_C_EC_only(b) == get_winner_loser(b)[0]):
#                    cluster_outcome_predictor += 1
#                if (not good2) and (armies_compositions_models[mu2].winner_battle_C_EC_only(b, True) == get_winner_loser(b)[0]):
#                    most_prob_cluster_outcome_predictor += 1
#                if (not good3) and (armies_compositions_models[mu2].winner_battle(b) == get_winner_loser(b)[0]):
#                    score_cluster_outcome_predictor += 1
#                if (not good4) and (armies_compositions_models[mu2].winner_battle(b, True) == get_winner_loser(b)[0]):
#                    score_most_prob_outcome_predictor += 1

            print "simple outcome predictor performance:", score_simple_outcome_predictor*1.0/len(test_battles), ':', score_simple_outcome_predictor, '/', len(test_battles)
            print "cluster only outcome predictor performance:", cluster_outcome_predictor*1.0/len(test_battles), ':', cluster_outcome_predictor, '/', len(test_battles)
            print "most prob cluster only outcome predictor performance:", most_prob_cluster_outcome_predictor*1.0/len(test_battles), ':', most_prob_cluster_outcome_predictor, '/', len(test_battles)
            print "(score * cluster factor) outcome predictor performance:", score_cluster_outcome_predictor*1.0/len(test_battles), ':', score_cluster_outcome_predictor, '/', len(test_battles)
            print "(score * most prob cluster) outcome predictor performance:", score_most_prob_outcome_predictor*1.0/len(test_battles), ':', score_most_prob_outcome_predictor, '/', len(test_battles)

    else:
        print >> sys.stderr, "usage:"
        print >> sys.stderr, "python armies.py [-d] [directory|file(s)] [-t]"
        print >> sys.stderr, "-d to load all the *.rgd files from a given directory"
        print >> sys.stderr, "-t to test / benchmark"


        from sklearn import decomposition
        from sklearn import mixture
        from sklearn import manifold
        from sklearn import cluster
        for race, list_of_percentages in armies_battles_for_clust.iteritems():
            print race
            compo = np.array(list_of_percentages)
            if len(compo):
                pca = decomposition.PCA()
                pca.fit(compo)
                print pca
                print pca.explained_variance_

                gmm = mixture.GMM(n_components=8, min_covar=0.000001, cvtype='full')
                gmm.fit(compo)
                print gmm
                print unit_types.by_race.military[race],
                print unit_types.by_race.drop[race]
                if WITH_STATIC_DEFENSE:
                    print unit_types.by_race.static_defense[race]
                print gmm.means

                dpgmm = mixture.DPGMM(cvtype='full')
                dpgmm.fit(compo)
                print dpgmm
                # print dpgmm.means
                # print dpgmm.precisions

                man = manifold.Isomap()
                man.fit(compo)
                print man
                print man.embedding_

                db = cluster.DBSCAN(eps=0.1)
                dbscan = db.fit(compo)
                print dbscan
                print dbscan.components_


            else:
                print "No battles"


        if SCORES_REGRESSION:
            armies_battles_regr_raw = np.array(armies_battles_for_regr, np.float32)
            armies_battles_regr_fscaled = data_tools.features_scaling(armies_battles_regr_raw)
            from sklearn import linear_model

            from sklearn.decomposition import PCA, FastICA
            from sklearn import cross_validation
            from sklearn import metrics

            X = armies_battles_regr_fscaled[:,:-1]
            Y = armies_battles_regr_fscaled[:,-1]
            clf = linear_model.LinearRegression()
            print clf.fit(X, Y)
            print clf.score(X, Y)
            scores = cross_validation.cross_val_score(
                         clf, X, Y, cv=5, score_func=metrics.euclidean_distances)
            print scores

            pca = PCA(1)
            X_r = pca.fit(X).transform(X)
            print clf.fit(X_r, Y)
            print clf.score(X_r, Y)
            scores = cross_validation.cross_val_score(
                         clf, X, Y, cv=5, score_func=metrics.euclidean_distances)
            print scores
            #pl.scatter(X_r, Y, color='black')
            #pl.plot(X_r, clf.predict(X_r), color='blue', linewidth=3)
            #pl.xticks(())
            #pl.yticks(())
            #pl.show()

            pca = PCA(2)
            X_r = pca.fit(X).transform(X)
            print clf.fit(X_r, Y)
            print clf.score(X_r, Y)
            scores = cross_validation.cross_val_score(
                         clf, X, Y, cv=5, score_func=metrics.euclidean_distances)
            print scores
            #pl.scatter(X_r[:,0], X_r[:,1], color='black')
            #pl.plot(X_r, clf.predict(X_r), color='blue', linewidth=3)
            #pl.xticks(())
            #pl.yticks(())
            #pl.show()

            clf = linear_model.BayesianRidge()
            print clf.fit(X, Y)
            print clf.score(X, Y)
            scores = cross_validation.cross_val_score(
                         clf, X, Y, cv=5, score_func=metrics.euclidean_distances)
            print scores

            clf = linear_model.Lasso(alpha = 0.1)
            print clf.fit(X, Y)
            print clf.score(X, Y)
            scores = cross_validation.cross_val_score(
                         clf, X, Y, cv=5, score_func=metrics.euclidean_distances)
            print scores

            from sklearn.linear_model import LassoCV, LassoLarsCV, LassoLarsIC

            clf = LassoLarsIC(criterion = 'bic')
            print clf.fit(X, Y)
            print clf.score(X, Y)
            scores = cross_validation.cross_val_score(
                         clf, X, Y, cv=5, score_func=metrics.euclidean_distances)
            print scores

            clf = LassoCV(cv=5)
            print clf.fit(X, Y)
            print clf.score(X, Y)
            scores = cross_validation.cross_val_score(
                         clf, X, Y, cv=5, score_func=metrics.euclidean_distances)
            print scores

            #import ldavb
            #clust = LDAVB()
            #print clust.fit(armies_battles_for_clust['P'])
            #print clust.fit(armies_battles_for_clust['T'])
            #print clust.fit(armies_battles_for_clust['Z'])
            
