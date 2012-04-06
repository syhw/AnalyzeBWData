class Unit:
    def __init__(self, unit_name, player, cdr, reg):
        self.name = unit_name
        self.player = player
        self.CDR = cdr
        self.Reg = reg

class GameState:
    def __init__(self):
        self.players_bases = {} # bases[player]['CDR'/'Reg'] = {where: number}
        self.floc = None
        self.last_loc_frame = -1
        self.tracked_units = {} # currently Command Centers/Nexii/Hatcheries

    def created(self, l):
        if 'Terran Command Center' in l or 'Protoss Nexus' in l or 'Zerg Hatchery' in l:
            # Morph doesn't change the unit id so no need for Lair/Hive
            l = l.split(',')
            self.tracked_units[int(l[3])] = Unit(l[4], l[1], 
                    int(l[-2]), int(l[-1]))
            self.add_base(self.tracked_units[int(l[3])])

    def destroyed(self, l):
        if 'Terran Command Center' in l or 'Protoss Nexus' in l or 'Zerg Hatchery' in l:
            l = l.split(',')
            u = self.tracked_units.pop(int(l[3]))
            self.remove_base(u)

    def remove_base(self, u):
        if u.player not in self.players_bases:
            print "ERROR player has no base and we have to remove one"
            return
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
        tmp = self.players_bases[u.player]
        tmp['CDR'][u.CDR] = tmp['CDR'].get(u.CDR, 0) + 1
        tmp['Reg'][u.Reg] = tmp['Reg'].get(u.Reg, 0) + 1

    def update_loc(self, uid, r, t='Reg'):
        self.remove_base(self.tracked_units[uid])
        if t == 'Reg':
            self.tracked_units[uid].Reg = r
        elif t == 'CDR':
            self.tracked_units[uid].CDR = r
        else:
            print "TYPE ERROR"
            return "TYPE ERROR"
        self.add_base(self.tracked_units[uid])

    def update(self , l):
        if 'Created' in l or 'Morph' in l:
            self.created(l)
        if 'Destroyed' in l:
            self.destroyed(l)
        if self.floc != None and len(l.split(',')) > 1:
            frame = int(l.split(',')[0])
            if frame > self.last_loc_frame: # only if frame increased
                for line in self.floc:
                    print line
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
        
