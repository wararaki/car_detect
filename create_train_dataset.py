'''
cnn train
'''
import os
import sys
import random, math
from PIL import Image
import numpy as np
from keras.preprocessing.image import img_to_array

def create_dataset(dir_path):
    '''
    create image data
    '''
    label_dirs = os.listdir(dir_path)
    # label_dirs.remove('train')
    # i => label
    train_data = []
    for i, label_dir in enumerate(label_dirs):
        file_path = dir_path + '/' + label_dir
        files = os.listdir(file_path)
        for _, file in enumerate(files):
            image_path = file_path + "/" + file
            img = Image.open(image_path)
            img = img.convert("RGB")
            data = np.asarray(img)
            train_data.append([data, i])
    # shuffle data
    random.shuffle(train_data)
    X, Y = [], []
    for data in train_data:
        X.append(data[0])
        Y.append(data[1])

    test_idx = math.floor(len(X) * 0.8)
    xy = (np.array(X[0:test_idx]), np.array(X[test_idx:]), 
          np.array(Y[0:test_idx]), np.array(Y[test_idx:]))
    np.save('./dataset/train.dat', xy)
    print(len(train_data))
    return train_data

def main():
    '''
    main function
    '''
    dataset_dir = "/Users/wararaki/dataset/yahoohack/train"
    create_dataset(dataset_dir)
    print("Done.")

    return 0

if __name__ == "__main__":
    sys.exit(main())
