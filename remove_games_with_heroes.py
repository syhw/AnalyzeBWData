import glob, sys, shutil
fnamelist = glob.iglob(sys.argv[1] + '/*.rgd')
to_remove = []
for fname in fnamelist:
    f = open(fname)
    header = True
    for line in f:
        if 'Created' in line and 'Hero' in line:
            to_remove.append(fname[:-3])
            break
    f.close()
#print to_remove
for fname in to_remove:
    shutil.move(fname+'rld', 'trash_games/'+fname.split('/')[-1]+'rld')
    shutil.move(fname+'rgd', 'trash_games/'+fname.split('/')[-1]+'rgd')
    shutil.move(fname+'rod', 'trash_games/'+fname.split('/')[-1]+'rod')
