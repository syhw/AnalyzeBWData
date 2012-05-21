from common import unit_types

go_to_2 = set([ # don't go to 11
    'Terran_Command_Center',
    'Terran_Supply_Depot',
    'Terran_Refinery',
    'Terran_Barracks',
    'Protoss_Nexus',
    'Protoss_Pylon',
    'Protoss_Assimilator',
    'Protoss_Gateway',
    'Zerg_Hatchery',
    'Zerg_Extractor',
    'Zerg_Overlord'])

go_to_3 = set([ # don't go to 11
    'Terran_Command_Center',
    'Terran_Supply_Depot',
    'Terran_Barracks',
    'Protoss_Nexus',
    'Protoss_Pylon',
    'Protoss_Gateway',
    'Zerg_Hatchery',
    'Zerg_Overlord'])

go_to_4 = set([ # don't go to 11
    'Terran_Barracks',
    'Protoss_Gateway',
    'Zerg_Hatchery']) 


class Unit:
    def __init__(self, uid, unit_name, player, dm=None, cdr=-1, reg=-1):
        self.uid = uid
        self.name = unit_name
        self.player = player
        self.CDR = -1
        self.Reg = -1
        if dm != None and type(dm) != int:
            if reg in dm.dist_Reg:
                self.Reg = reg
            if cdr in dm.dist_CDR:
                self.CDR = cdr
    def __getitem__(self, x):
        if x == 'CDR':
            return self.CDR
        elif x == 'Reg':
            return self.Reg
        else:
            print x
            raise TypeError

class GameState:
    """
    Keeps track of alive units and their positions, keeps tracks of bases, 
    tracked_units/remove/add could be refactored: tracked_u[uid] = Unit(uid,...
    """
    def __init__(self, dm=None):
        self.players_races = {} # races[player] in {'P', 'T', 'Z'}
        self.players_bases = {} # bases[player]['CDR'/'Reg'] = {where: number}
        self.floc = None
        self.last_loc_frame = -1
        self.tracked_units = {} 
        self.base_uid = set()
        self.dm = dm

    def created(self, line):
        l = line.split(',')
        uid = int(l[3])
        if not l[1] in self.players_races:
            if 'Protoss' in line:
                self.players_races[l[1]] = 'P'
            elif 'Terran' in line:
                self.players_races[l[1]] = 'T'
            elif 'Zerg' in line:
                self.players_races[l[1]] = 'Z'
        if len(l) < 9 or self.dm == None:
            self.tracked_units[uid] = Unit(uid, l[4], l[1])
        else:
            self.tracked_units[uid] = Unit(uid, l[4], l[1], self.dm,
                    int(l[-2]), int(l[-1]))
        if 'Terran Command Center' in line or 'Protoss Nexus' in line or 'Zerg Hatchery' in line:
            self.add_base(self.tracked_units[uid])

    def morphed(self, line):
        l = line.split(',')
        uid = int(l[3])
        if len(l) < 9 or self.dm == None:
            if uid in self.tracked_units and self.dm != None:
                self.tracked_units[uid] = Unit(uid, l[4], l[1], self.dm,
                    self.tracked_units[uid].CDR, self.tracked_units[uid].Reg)
            else:
                self.tracked_units[uid] = Unit(uid, l[4], l[1])
        else:
            self.tracked_units[uid] = Unit(uid, l[4], l[1], self.dm,
                    int(l[-2]), int(l[-1]))
        if 'Zerg Hatchery' in line: #or 'Zerg Lair' in line or 'Zerg Hive' in line:
            self.add_base(self.tracked_units[uid])

    def destroyed(self, line):
        l = line.split(',')
        u = self.tracked_units.pop(int(l[3]))
        if 'Terran Command Center' in line or 'Protoss Nexus' in line or 'Zerg Hatchery' in line or 'Zerg Lair' in line or 'Zerg Hive' in line:
            self.remove_base(u)

    def remove_base(self, u):
        if u.player not in self.players_bases:
            print "ERROR player has no base and we have to remove one"
            return
        self.base_uid.remove(u.uid)
        tmp = self.players_bases[u.player]
        tmp['CDR'][u.CDR] = tmp['CDR'].get(u.CDR, 0) - 1
        tmp['Reg'][u.Reg] = tmp['Reg'].get(u.Reg, 0) - 1
        if tmp['CDR'][u.CDR] < 1:
            tmp['CDR'].pop(u.CDR)
        if tmp['Reg'][u.Reg] < 1:
            tmp['Reg'].pop(u.Reg)

    def add_base(self, u):
        if u.player not in self.players_bases:
            self.players_bases[u.player] = {'CDR': {}, 'Reg': {}}
        self.base_uid.add(u.uid)
        tmp = self.players_bases[u.player]
        tmp['CDR'][u.CDR] = tmp['CDR'].get(u.CDR, 0) + 1
        tmp['Reg'][u.Reg] = tmp['Reg'].get(u.Reg, 0) + 1

    def update_loc(self, uid, r, t='Reg'):
        if uid in self.base_uid and self.players_races[self.tracked_units[uid].player] == 'T':
            # we want only moving CC to be removed as base
            self.remove_base(self.tracked_units[uid])
        if t == 'Reg' and r in self.dm.dist_Reg:
            self.tracked_units[uid].Reg = r
        elif t == 'CDR' and r in self.dm.dist_CDR:
            self.tracked_units[uid].CDR = r
        if 'Terran Command Center' in self.tracked_units[uid].name:
            # it should be when it lands...
            self.add_base(self.tracked_units[uid])

    def update(self , l):
        if 'Created' in l:
            self.created(l)
        if 'Morph' in l:
            self.morphed(l)
        if 'Destroyed' in l:
            self.destroyed(l)
        if self.floc != None and len(l.split(',')) > 1:
            frame = int(l.split(',')[0])
            if frame > self.last_loc_frame: # only if frame increased
                for line in self.floc:
                    t = ''
                    if 'CDR' in line:
                        t = 'CDR'
                    elif 'Reg' in line:
                        t = 'Reg'
                    if t != '':
                        li = line.rstrip('\r\n').split(',')
                        if len(li) < 2:
                            break
                        if li[1] != '' and li[3] != '':
                            unitmp = int(li[1]) 
                            regtmp = int(li[3]) 
                            if unitmp in self.tracked_units: #and regtmp in 
                                self.update_loc(unitmp, regtmp, t)
                        if int(li[0]) >= frame: # TODO not always good
                            break               # RGD can go faster than RLD
                self.last_loc_frame = frame     # ~

    def track_loc(self, f):
        self.floc = f
        for line in self.floc: # positions the reader (seek()) pointer at start
            if '[Replay Start]' in line:
                break
        
    def has_one_of(self, units, player):
        for u in self.tracked_units.itervalues():
            if u.player == player and u.name in units:
                return True
        return False

    def has_all_of(self, units, player):
        has_units = set([u.name for u in self.tracked_units.itervalues() if u.player==player])
        for u in units:
            if u not in has_units:
                return False
        return True

    def get_buildings(self, player):
        r = set()
        for u in self.tracked_units.itervalues():
            if u.player == player and\
                u.name in unit_types.buildings_sets[self.players_races[player]]:
                un = unit_types.map_to_TT_enum(u.name)
                if un == '':
                    continue
                if un in r and un in go_to_2:
                    if un+'2' in r and un in go_to_3:
                        if un+'3' in r and un in go_to_4:
                            r.add(un+'4')
                        else:
                            r.add(un+'3')
                    else:
                        r.add(un+'2')
                else:
                    r.add(un)
        return r


