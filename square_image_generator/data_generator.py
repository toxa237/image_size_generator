import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import keras


class SheetImageGenerator(keras.utils.PyDataset):
    def __init__(self, directory, batch_size=33, start_shape=(200, 200), out_size=(50, 50),
                 pises_count=4, **kwargs):
        super().__init__(**kwargs)
        self.directory = directory
        self.batch_size = batch_size
        self.start_shape = np.array(start_shape)
        self.out_size = np.array(out_size)
        self.files = os.listdir(directory)
        self.num_samples = len(self.files)
        self.piece_count = pises_count

    def __len__(self):
        return self.num_samples

    def __getitem__(self, item):
        file_pash = os.path.join(self.directory, self.files[item])
        image = tf.keras.utils.img_to_array(tf.keras.utils.load_img(file_pash))



if __name__ == '__main__':
    gen = SheetImageGenerator('data/data2', batch_size=12, piece_size=(200, 200))
    x, y = gen[0]
    print(x.shape)
    for i in range(5):
        plt.figure()
        plt.subplot(1, 4, 1)
        plt.imshow(x[i, 0])
        plt.subplot(1, 4, 2)
        plt.imshow(x[i, 1])
        plt.subplot(1, 4, 3)
        plt.imshow(x[i, 2])
        plt.subplot(1, 4, 4)
        plt.imshow(y[i])
    plt.show()
