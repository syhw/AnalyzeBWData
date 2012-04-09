class Unit:
    def __init__(self, uid, unit_name, player, cdr=-1, reg=-1):
        self.uid = uid
        self.name = unit_name
        self.player = player
        self.CDR = cdr
        self.Reg = reg
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
    def __init__(self):
        self.players_bases = {} # bases[player]['CDR'/'Reg'] = {where: number}
        self.floc = None
        self.last_loc_frame = -1
        self.tracked_units = {} 
        self.base_uid = set()

    def created(self, line):
        l = line.split(',')
        uid = int(l[3])
        if len(l) < 9:
            self.tracked_units[uid] = Unit(uid, l[4], l[1])
        else:
            self.tracked_units[uid] = Unit(uid, l[4], l[1], 
                    int(l[-2]), int(l[-1]))
        if 'Terran Command Center' in line or 'Protoss Nexus' in line or 'Zerg Hatchery' in line:
            self.add_base(self.tracked_units[uid])

    def morphed(self, line):
        l = line.split(',')
        uid = int(l[3])
        if len(l) < 9:
            if uid in self.tracked_units:
                self.tracked_units[uid] = Unit(uid, l[4], l[1],
                    self.tracked_units[uid].CDR, self.tracked_units[uid].Reg)
            else:
                self.tracked_units[uid] = Unit(uid, l[4], l[1])
        else:
            self.tracked_units[uid] = Unit(uid, l[4], l[1], 
                    int(l[-2]), int(l[-1]))
        if 'Zerg Hatchery' in line:
            self.add_base(self.tracked_units[uid])

    def destroyed(self, line):
        l = line.split(',')
        u = self.tracked_units.pop(int(l[3]))
        if 'Terran Command Center' in line or 'Protoss Nexus' in line or 'Zerg Hatchery' in line:
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
        if uid in self.base_uid:
            self.remove_base(self.tracked_units[uid])
        if t == 'Reg':
            self.tracked_units[uid].Reg = r
        elif t == 'CDR':
            self.tracked_units[uid].CDR = r
        else:
            print "TYPE ERROR"
            return "TYPE ERROR"
        if 'Terran Command Center' in self.tracked_units[uid].name:
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
                        if int(li[1]) in self.tracked_units:
                            self.update_loc(int(li[1]), int(li[3]), t)
                        if int(li[0]) >= frame: # TODO not always good
                            break               # RGD can go faster than RLD
                self.last_loc_frame = frame     # ~

    def track_loc(self, f):
        self.floc = f
        for line in self.floc: # positions the reader (seek()) pointer at start
            if '[Replay Start]' in line:
                break
        