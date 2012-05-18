#!/usr/bin/python
# -*- coding: utf-8 -*-
# TODO clustering to try:
#  - roll our own (see ArmyCompositions)
#  - PCA fragmentation on the first principal component
#  - PCA fragmentation on first few principal components
#  - Linear Discriminant Analysis same as above
#  - GMM EM (which distance function w.r.t. the 0 !!= 0.01 problem)
#  - GMM DP (which distance function w.r.t. the 0 !!= 0.01 problem)
#  - Latent Dirichlet Analysis

import sys, os, pickle, copy, itertools, functools, math, random
from collections import defaultdict
from common import data_tools
from common import unit_types
from common import attack_tools
from common.vector_X import *
from common.common_tools import memoize
try:
    import numpy as np
except:
    print "you need numpy"
    sys.exit(-1)

SCORES_REGRESSION = False # try to do battles scores regressions
MIN_POP_ENGAGED = 6 # 12 zerglings, 6 marines, 3 zealots
MAX_FORCES_RATIO = 1.5 # max differences between engaged forces
WITH_STATIC_DEFENSE = False # tells if we include static defense in armies
CSV_ARMIES_OUTPUT = True # CSV output of the armies compositions
DEBUG_OUR_CLUST = True # debugging output for our clustering
NUMBER_OF_TEST_GAMES = 10 # number of test games to use
PARALLEL_COORDINATES_PLOT = False # should we plot units percentages?
width = 0.01 # width of bins in P(Unit_i | C)
epsilon = 1.0e-16 # lowest not zero

print >> sys.stderr, "SCORES_REGRESSION ",    SCORES_REGRESSION 
print >> sys.stderr, "MIN_POP_ENGAGED ",      MIN_POP_ENGAGED 
print >> sys.stderr, "MAX_FORCES_RATIO ",     MAX_FORCES_RATIO 
print >> sys.stderr, "WITH_STATIC_DEFENSE ",  WITH_STATIC_DEFENSE 
print >> sys.stderr, "CSV_ARMIES_OUTPUT ",    CSV_ARMIES_OUTPUT 
print >> sys.stderr, "DEBUG_OUR_CLUST ",      DEBUG_OUR_CLUST 
print >> sys.stderr, "NUMBER_OF_TEST_GAMES ", NUMBER_OF_TEST_GAMES 
print >> sys.stderr, "PARALLEL_COORDINATES_PLOT ", PARALLEL_COORDINATES_PLOT 

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
        s = sum(v.itervalues())
        tmp = {}
        for unit, numbers in v.iteritems():
            tmp[unit] = 1.0*numbers/s
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
    for line in f:
        obs.detect_observers(line)
        if 'IsAttacked' in line:
            tmp = data_tools.parse_dicts(line, lambda x: int(x))
            # sometimes the observers are detected in the fight (their SCVs)
            tmp = obs.heuristics_remove_observers(tmp)
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
    if pop_max[p1] > MIN_POP_ENGAGED*2 and pop_max[p2] > MIN_POP_ENGAGED*2 and score_before[p1] < MAX_FORCES_RATIO*score_before[p2] and score_before[p2] < MAX_FORCES_RATIO*score_before[p1]:
        compo = to_ratio(armies_battle[1])
        score_after = score_units(armies_battle[2])
        return compo[p1], compo[p2], score_after[p1], score_after[p2]
    else:
        return [], [], [], []

def format_battle_for_clust(players_races, armies_battle):
    """ take an "extract_armies_battles" formatted battle data and make it
    ready for clustering (ex: 'Protoss Dragoon': 12, 'Protoss Zealot': 6)
    returning 1 dict of vectors of units numbers per unit types, per race """
    r = {'P': [], 'T': [], 'Z': []}
    for k,v in players_races.iteritems():
        if k in armies_battle[1]: # armies_battle[1] is "max units engaged"
            r[v].append(copy.deepcopy(armies_battle[1][k]))
    return r

class ArmyCompositions:
    ut_by_race = {}
    for race in ['T', 'P', 'Z']:
        ut_by_race[race] = unit_types.by_race.military[race]+[unit_types.by_race.drop[race]]
        if WITH_STATIC_DEFENSE:
            ut_by_race[race].extend(unit_types.by_race.static_defense[race])

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
                    'T_vulture': {'Terran Vulture': 0.3},
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
                return width*1.0/(max_p-min_p) - (1.0-max_p+ min_p)*epsilon
            else:
                return epsilon
        for name,compo in self.compositions.iteritems():
            for unit,min_percentage in compo.iteritems():
                max_percentage = 1.0-sum([v for k,v in compo.iteritems() if k != unit])
                if unit not in self.P_unit_knowing_cluster:
                    self.P_unit_knowing_cluster[unit] = defaultdict(lambda: lambda x: width)
                self.P_unit_knowing_cluster[unit].update({name: functools.partial(f, min_percentage, max_percentage)})
        ArmyCompositions.ac_by_race[self.race] = self

    def distrib(self, percents_list):
        """ Computes ∏_i P(U_i|C=c) ∀ clusters c in C, returns {c: logprob} """
        d = {}
        for cluster in self.compositions:
            d[cluster] = 0.0
            for i, unit_type in enumerate(ArmyCompositions.ut_by_race[self.race]):
                d[cluster] += math.log(self.P_unit_knowing_cluster[unit_type][cluster](percents_list[i]))
        return d

    def count(self, d):
        """ Adds log probabilities of clusters for a given d """
        for c, logprob in d.iteritems():
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
            #print "removing cluster", clust
            self.compositions.pop(clust)
            for unit in self.P_unit_knowing_cluster:
                self.P_unit_knowing_cluster[unit].pop(clust, 0)
        print "Clusters:"
        for clust in self.compositions:
            print clust


class ArmyCompositionModel:
    @staticmethod
    @memoize
    def unit_to_int(unit):
        return ArmyCompositions.ut_by_race[unit[0]].index(unit)

    @staticmethod
    @memoize
    def cluster_to_int(cluster):
        return [k for k in ArmyCompositions.ac_by_race[cluster[0]].compositions].index(cluster)

    @staticmethod
    @memoize
    def int_to_cluster(i):
        return [k for k in ArmyCompositions.ac_by_race[cluster[0]].compositions][i]

    def Cfinal_knowing_CtacticsCcounter(self, tt, P_Ctactics, P_Ccounter):
        # Which values of C are compatible with tt? (set of buildings we have)
        def has_all_requirements(list_ut):
            for ut in list_ut:
                for req in unit_types.required_for(ut):
                    if req not in tt:
                        return False
            return True
        tmp = []
        for i in range(len(self.Ccounter_knowing_ECnext)):
            cluster = ArmyCompositionModel.int_to_cluster(i)
            if has_all_requirements(ArmyCompositions.ac_by_race[self.race].compositions[cluster].keys()):
                tmp.append(0.5)
            else:
                tmp.append(0.0)
        d1 = np.array(tmp)
        # Fusion values for Ctactics and Ccounter
        d2 = self.alpha*P_Ctactics + (1-self.alpha)*P_Ccounter
        return d1*d2 # TODO verify
    
    def __init__(self, ac, eac, ett, alpha=0.25):
        """
        Takes two ArmyComposition objects (for us and for the enemy) and
        two vector_X objects (techtree for us and for the enemy)
        and builds an ArmyCompositionModel
        """
        self.alpha = alpha
        self.race = ac.race
        self.erace = eac.race
        self.matchup = self.race + 'v' + self.erace
        range_discretization = range(int(1.0/width))
        self.EU_knowing_EC = np.ndarray(shape=(len(range_discretization),
            len(eac.P_unit_knowing_cluster),
            len(eac.compositions)), dtype='float')
        self.EC_knowing_ECnext = np.ndarray(shape=(len(eac.compositions),
            len(eac.compositions)), dtype='float')
        self.ECnext_knowing_TT = np.ndarray(shape=(len(eac.compositions),
            len(ett.vector_X)), dtype='float')
        self.Ccounter_knowing_ECnext = np.ndarray(shape=(len(ac.compositions),
            len(eac.compositions)), dtype='float')
        self.U_knowing_Cfinal = np.ndarray(shape=(len(range_discretization),
            len(ac.P_unit_knowing_cluster),
            len(ac.compositions)), dtype='float')
        for unit in ac.P_unit_knowing_cluster:
            for cluster, prob in ac.P_unit_knowing_cluster[unit].iteritems():
                for i in range_discretization:
                    self.U_knowing_Cfinal[i][ArmyCompositionModel.unit_to_int(unit)][ArmyCompositionModel.cluster_to_int(cluster)] = prob(i)
        for unit in eac.P_unit_knowing_cluster:
            for cluster, prob in eac.P_unit_knowing_cluster[unit].iteritems():
                for i in range_discretization:
                    self.EU_knowing_EC[i][ArmyCompositionModel.unit_to_int(unit)][ArmyCompositionModel.cluster_to_int(cluster)] = prob(i)

    def train(self, battle):
        get_players_race(battle)
        get_winner(battle)


class percent_list(list):
    """ a type of list which adds percentages of units types in units order """
    def new_battle(self, d):
        race = d.iterkeys().next()[0] # first character of the first unit
        tmp = [d.get(u, 0.0) for u in ArmyCompositions.ut_by_race[race]]
        if race == 'T':
            tmp[unit_types.by_race.military[race].index('Terran Siege Tank Tank Mode')] += tmp.pop(unit_types.by_race.military[race].index('Terran Siege Tank Siege Mode'))
        self.append(tmp)


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


def get_players_race(b):
    # b[0] = army of player 0, b[1] = army of player 1
    r = []
    for i in [0,1]:
        first_unit = b[i].iterkeys().next()
        r.append(first_unit[0])
    return r


def get_winner(b):
    # b[-2] = score player 0, b[-1] = score player 1
    if b[-1] > b[-2]:
        return 1
    else:
        return 0


f = sys.stdin
armies_battles_for_regr = []
armies_battles_for_clust = {'P': percent_list(), 
        'T': percent_list(), 
        'Z': percent_list()}
battles_for_clustering = []
fnamelist = []
ArmyCompositions('P')
ArmyCompositions('T')
ArmyCompositions('Z')
armies_compositions_models = {}

if __name__ == "__main__":
    if len(sys.argv) > 1:
        if sys.argv[1] == '-d': # -d for directory
            import glob
            fnamelist = glob.iglob(sys.argv[2] + '/*.rgd')
        else:
            fnamelist = sys.argv[1:]
        learngames = [fna for fna in fnamelist]
        testgames = []
        if '-t' in sys.argv: # -t for tests
            if NUMBER_OF_TEST_GAMES > len(fnamelist):
                print >> sys.stderr, "Number of test games > number of games"
                sys.exit(-1)
            i = 0
            random.seed(0) # no randomness, just sampling
            r = random.randint(0, len(fnamelist)-1)
            while i < NUMBER_OF_TEST_GAMES and i < 5000000:
                if fnamelist[r] not in testgames:
                    i += 1
                    testgames.append(fnamelist[r])
                    learngames.remove(fnamelist[r])
                r = random.randint(0,len(fnamelist)-1)

        print "learning from", len(learngames), "games"
        for fname in learngames:
            f = open(fname)
            players_races = data_tools.players_races(f)

            ### Parse battles and extract armies (before, after)
            armies_raw = extract_armies_battles(f)

            if SCORES_REGRESSION:
                ### Format battles for predict/regression (of the outcome)
                battles_r = map(functools.partial(format_battle_for_regr,
                        players_races), armies_raw)
                armies_battles_for_regr.extend(battles_r)

            ### Format battles for clustering (with armies order P>T>Z)
            ### (army_p1, army_p2, final_score_p1, final_score_p2)
            battles_c = filter(lambda x: len(x[0]) and len(x[1]),
                    map(functools.partial(format_battle_for_clust_adv,
                    players_races), armies_raw))
            #print battles_c

            ### save these battles for further use
            battles_for_clustering.extend(battles_c)

            ### Sort armies by race and put inside battles_for_clust
            for b in battles_c:
                assert(len(b) == 4) # enforce that b is a battle in the format
                # (army_p1, army_p2, score_p1, score_p2)
                for race in armies_battles_for_clust.iterkeys():
                    for i,prace in enumerate(get_players_race(b)):
                        if prace == race:
                            armies_battles_for_clust[race].new_battle(b[i])
            #print armies_battles_for_clust

        if PARALLEL_COORDINATES_PLOT:
            from common.parallel_coordinates import parallel_coordinates
        annotated_l = []
        for race, l in armies_battles_for_clust.iteritems():
            if len(l) > 0:
                armies_compositions_models[matchup(race, armies_battles_for_clust)] = 0
                x_l = [u for u in ArmyCompositions.ut_by_race[race]]
                if race == 'T':
                    x_l.pop(unit_types.by_race.military[race].index('Terran Siege Tank Siege Mode'))
                x_l.append('MostProbableClust')
                x_l = map(lambda s: ''.join(s.split(' ')[1:]), x_l)
                if PARALLEL_COORDINATES_PLOT:
                    parallel_coordinates(l, x_labels=x_l).savefig("parallel_"+race+".png")
                for p_l in l:
                    dist = ArmyCompositions.ac_by_race[race].distrib(p_l) 
                    ArmyCompositions.ac_by_race[race].count(dist)
                    decreasing_probs_clusters = sorted([(c,logprob) for c,logprob in dist.iteritems()], key=lambda x: x[1], reverse=True)
                    if DEBUG_OUR_CLUST:
                        print zip(x_l, map(lambda x: "%.2f" % x, p_l))
                        print decreasing_probs_clusters[:5]
                    annotated_l.append(p_l + [decreasing_probs_clusters[0][0]])
                    # /!\ annotated without pruning (goes after)
                ArmyCompositions.ac_by_race[race].prune()

                if CSV_ARMIES_OUTPUT:
                    csv = open(race+'_armies.csv', 'w')
                    csv.write(','.join(x_l) + '\n')
                    for line in annotated_l:
                        csv.write(','.join(map(lambda e: str(e), line))+'\n')

        for mu in armies_compositions_models:
            armies_compositions_models[mu] = ArmyCompositionModel(ArmyCompositions.ac_by_race[mu[0]], ArmyCompositions.ac_by_race[mu[2]], vector_X(mu[0], mu[2]), vector_X(mu[2], mu[0])) # TODO review
            for battle in battles_for_clustering:
                # (army_p1, army_p2, final_score_p1, final_score_p2)
                armies_compositions_models[mu].train(battle)
                


        for fname in testgames:
            f = open(fname)
            players_races = data_tools.players_races(f)

            ### Parse battles and extract armies (before, after)
            armies_raw = extract_armies_battles(f)

            if SCORES_REGRESSION:
                ### Format battles for predict/regression (of the outcome)
                battles_r = map(functools.partial(format_battle_for_regr,
                        players_races), armies_raw)
                armies_battles_for_regr.extend(battles_r)

            ### Format battles for clustering (with armies order P>T>Z)
            ### (army_p1, army_p2, final_score_p1, final_score_p2)
            battles_c = filter(lambda x: len(x[0]) and len(x[1]),
                    map(functools.partial(format_battle_for_clust_adv,
                    players_races), armies_raw))
            #print battles_c

            ### save these battles for further use
            battles_for_clustering.extend(battles_c)

            ### Sort armies by race and put inside battles_for_clust
            for b in battles_c:
                for race in armies_battles_for_clust.iterkeys():
                    first_unit = b[0].iterkeys().next()
                    if first_unit[0] == race:
                        armies_battles_for_clust[race].new_battle(b[0])
                    first_unit = b[1].iterkeys().next()
                    if first_unit[0] == race:
                        armies_battles_for_clust[race].new_battle(b[1])
            #print armies_battles_for_clust
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

            import pylab as pl
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
            
