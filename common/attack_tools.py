
class Observers:
    def __init__(self):
        self.obs = []
        self.num_created = {}

    def detect_observers(self, line):
        if len(self.obs) == 0 and 'Created' in line:
            l = line.split(',')
            if l[1] not in self.num_created:
                self.num_created[l[1]] = 1
            else:
                self.num_created[l[1]] += 1
            if int(l[0]) > 4320: # 3 minutes * 60 seconds * 24 frames/s
                for k,v in self.num_created.iteritems():
                    if v < 8: # 5 workers + 1 townhall = 6 created by starting
                        self.obs.append(k)

    def heuristics_remove_observers(self, d):
        """ look if there are more than 2 players engaged in the battle
        and seek player with nothing else than SCV and Command Centers engaged
        in the battle
        """
        if len(d[0]) > 2: # d[0] are all units involved, if len(d[0] > 2
            # it means that there are more than 2 players in the battle
            if len(d[0]) - len(self.obs) > 2: # a player is not captured by the obs
                # heuristic (building SCV at the beginning)
                keys_to_del = set()
                for k,v in d[0].iteritems():
                    to_del = True
                    for unit in v:
                        if unit != 'Terran SCV' and unit != 'Terran Command Center':
                            to_del = False
                            break
                    if to_del:
                        keys_to_del.add(k)
                for kk in range(len(d)):
                    for k in keys_to_del:
                        d[kk].pop(k)
            else: # remove the observing players from the battle
                for kk in range(len(d)):
                    for k in self.obs:
                        d[kk].pop(k)
        return d
