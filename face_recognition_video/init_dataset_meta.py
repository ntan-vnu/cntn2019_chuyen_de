import glob
import random

def save_meta_file(files, filename):
    with open(filename, 'wt') as f:
        # 'data/<uid>/...
        for it in files:
            uid = it.split('/')[1]
            f.write('%s\t%s\n'%(it, uid))
    pass

if __name__ == "__main__":
    files = glob.glob('data/*/*')
    random.shuffle(files)

    n_train = int(0.7*len(files))
    f_trains = files[:n_train]
    f_tests = files[n_train:]
    save_meta_file(f_trains, 'train.txt')
    save_meta_file(f_tests, 'test.txt')
    pass