import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import keras


class SheetImageGenerator(keras.utils.PyDataset):
    def __init__(self, directory, batch_size=33, piece_size=(50, 50), pises_count=4, **kwargs):
        super().__init__(**kwargs)
        self.directory = directory
        self.batch_size = batch_size
        self.piece_size = np.array(piece_size)
        self.files = os.listdir(directory)
        self.num_samples = len(self.files)
        self.piece_count = pises_count

    def __len__(self):
        return self.num_samples

    def __getitem__(self, item):
        file_pash = os.path.join(self.directory, self.files[item])
        image = keras.utils.img_to_array(keras.utils.load_img(file_pash)) / 255
        max_h = image.shape[0] - 4 * self.piece_size[0]
        max_w = image.shape[1] - 4 * self.piece_size[1]

        batch_input = []
        batch_output = []
        for _ in range(self.batch_size):
            vector = np.random.choice([self._get_vertical_line, self._get_horizontal_line])
            images_input = vector(image, max_w, max_h)
            images_input = images_input[::np.random.choice([-1, 1])]
            images_output = images_input.pop(-1)
            batch_input.append(images_input)
            batch_output.append(images_output)
        batch_input_dis, batch_input_gen  = np.split(np.array(batch_input), [self.batch_size//3], axis=0)
        batch_output_dis = np.array(batch_output)[:self.batch_size//3]
        return (batch_input_dis, batch_input_gen), batch_output_dis

    def _get_vertical_line(self, image, max_w, max_h):
        x_cord = np.random.randint(max_w)
        y_cord = np.random.randint(max_h)
        images_input = image[x_cord: x_cord + self.piece_size[0] * self.piece_count,
                             y_cord:y_cord + self.piece_size[0]]
        images_input = np.split(images_input, self.piece_count, axis=0)
        return images_input

    def _get_horizontal_line(self, image, max_w, max_h):
        x_cord = np.random.randint(max_w)
        y_cord = np.random.randint(max_h)
        images_input = image[x_cord: x_cord + self.piece_size[0],
                             y_cord:y_cord + self.piece_size[0] * self.piece_count]
        images_input = np.split(images_input, self.piece_count, axis=1)
        return images_input


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
