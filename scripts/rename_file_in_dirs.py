import os
path = '/home/niu/Documents/Datesets/calcucate_dirs/UVB_DRIT'
dirs = os.listdir(path)
for i, dir in enumerate(dirs):
    # print('==============')
    # print(dir)
    p = os.path.join(path, '{}'.format(str(dir)))
    # print(p)
    # print('--------------')
    d = os.listdir(p)
    for j, file in enumerate(d):
        NewName = os.path.join(p,'{}_{}.png'.format(str(dir),str(j)))
        OldName = os.path.join(p, file)
        # print(NewName)
        # print(OldName)
        # NewName = os.path.join(path, "{}".format(str(j))+file)
        # OldName = os.path.join(path, file)
        os.rename(OldName, NewName)
