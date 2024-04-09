import glob 
import os 
import statistics


def remove_previous_files(path):
    existing_files = glob.glob(f'{path}*')
    for f in existing_files:
        os.remove(f)
    return

def write_accuracy_to_file(file_path, accs):
    file_path = os.path.join(file_path, 'acc.txt')
    with open(file_path, 'w+') as f:
        for i, acc in enumerate(accs):
            f.write(f'Fold {i+1}: {acc}\n')
        f.write(f'Mean: {statistics.mean(accs)}\n')
    f.close()
    return