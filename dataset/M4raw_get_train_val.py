import glob
paths = glob.glob('/home/jupyter-huangshoujin/shoujinhuang/data/M4RawV1.1/multicoil_train/*T1*')
f = open('M4Raw_train_split_less.csv','w')
for path in paths:
    T1 = path.split('/')[-1].replace('.h5','')
    T2 = T1.replace('T1','T2').replace('.h5','')
    f.write(f'{T1},{T2}\n')
f.close()
paths = glob.glob('/home/jupyter-huangshoujin/shoujinhuang/data/M4RawV1.1/multicoil_val/*T1*')
f = open('M4Raw_val_split_less.csv','w')
for path in paths:
    T1 = path.split('/')[-1].replace('.h5','')
    T2 = T1.replace('T1','T2').replace('.h5','')
    f.write(f'{T1},{T2}\n')
f.close()