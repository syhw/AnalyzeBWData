import sys
from common import data_tools
from common.position_tools import PositionMapper
from common.position_tools import DistancesMaps
from battles import tactics

fnamelist = []
if sys.argv[1] == '-d':
    import glob
    for g in glob.iglob(sys.argv[2] + '/*.rgd'):
        fnamelist.append(g)
else:
    fnamelist = [fnam for fnam in sys.argv[1:] if fnam[0] != '-']

mean_tactics_different_regions = {'Reg': 0.0, 'CDR': 0.0}

for fname in fnamelist:
    print fname
    f = open(fname)
    floc = open(fname[:-3]+'rld')
    dm = DistancesMaps(floc)
    floc.close()
    pm = PositionMapper(dm, fname[:-3])
    pr = data_tools.players_races(f)
    battles = tactics.extract_tactics_battles(fname, pr, dm, pm)
    game_regs = {'Reg': set(), 'CDR': set()}
    for battle in battles:
        for rt in ['Reg', 'CDR']:
            game_regs[rt].add(battle[-1][rt])
    for rt in ['Reg', 'CDR']:
        mean_tactics_different_regions[rt] += len(game_regs[rt])

for rt in ['Reg', 'CDR']:
    mean_tactics_different_regions[rt] /= len(fnamelist)

print mean_tactics_different_regions

