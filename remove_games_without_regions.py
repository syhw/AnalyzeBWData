import glob, sys, shutil
fnamelist = glob.iglob(sys.argv[1] + '/*.rld')
to_remove = []
for fname in fnamelist:
    f = open(fname)
    header = True
    gotCDR = False
    gotReg = False
    for line in f:
        if '[Replay Start]' in line: 
            header = False
        if not header:
            if 'CDR' in line:
                gotCDR = True
            if 'Reg' in line:
                gotReg = True
    if not gotCDR or not gotReg:
        to_remove.append(fname[:-3])
    f.close()
#print to_remove
#d = sys.argv[1].split('/')[:-1]
#print d
for fname in to_remove:
    shutil.move(fname+'rld', 'trash_games/'+fname.split('/')[-1]+'rld')
    shutil.move(fname+'rgd', 'trash_games/'+fname.split('/')[-1]+'rgd')
    shutil.move(fname+'rod', 'trash_games/'+fname.split('/')[-1]+'rod')
