import numpy as np
import csv
import glob, os
from tqdm import tqdm
from shutil import copyfile
import cv2

pathImageNet = './DataSets/ImageNet/'
pathminiImageNet = './DataSets/miniImageNet/allimages'
targetpath = './DataSets/miniImageNet/miniImages'
resizetargetpath = './DataSets/miniImageNet/resizedminiImages'
csv_file_dir = './miniImagenet'
filesCSVSachinRavi = [os.path.join(csv_file_dir, 'train.csv'),
                      os.path.join(csv_file_dir, 'val.csv'),
                      os.path.join(csv_file_dir, 'test.csv')]

for filename in filesCSVSachinRavi:
    with open(filename) as csvfile:
        csv_reader = csv.reader(csvfile, delimiter=',')
        next(csv_reader, None)
        images = {}
        print('Reading IDs....')
        for row in tqdm(csv_reader):
            if row[1] in images.keys():
                images[row[1]].append(row[0])
            else:
                images[row[1]] = [row[0]]

        print('Writing photos....')
        for c in tqdm(images.keys()): # Iterate over all the classes
            lst_files = []
            for file in glob.glob(pathminiImageNet + "/*"+c+"*"):
                lst_files.append(file) # the absolute path
            # TODO: Sort by name of by index number of the image???
            # I sort by the number of the image
            lst_index = [int(i[i.rfind('_')+1:i.rfind('.')]) for i in lst_files]
            index_sorted = sorted(range(len(lst_index)), key=lst_index.__getitem__)
            # print(images[c])
            #
            # # Now iterate
            index_selected = [int(i[i.index('.') - 4:i.index('.')]) for i in images[c]]
            selected_images = np.array(index_sorted)[np.array(index_selected) - 1]
            # print("selected Images", selected_images)
            for i in np.arange(len(selected_images)):
                # read file and resize to 84x84x3
                im = cv2.imread(os.path.join(pathImageNet,lst_files[selected_images[i]]))
                im_resized = cv2.resize(im, (84, 84), interpolation=cv2.INTER_AREA)
                cv2.imwrite(os.path.join(resizetargetpath, images[c][i]), im_resized)
                copyfile(os.path.join(pathminiImageNet, lst_files[selected_images[i]]), os.path.join(targetpath, images[c][i]))
