import numpy as np
import matplotlib.pyplot as plt
from keras.api.models import load_model
from keras.api.utils import load_img


class ImageResizer:
    def __init__(self):
        self.model = load_model('models/model50_50')
        self.size = 50

    def predict(self, img, left=0, right=0, top=0, bottom=0):
        if right:
            for i in range(right//self.size+1):
                img = self.predict_vertical_right(img)
            img = img[:, :-(self.size - right % self.size)]
        if left:
            for i in range(left//self.size+1):
                img = self.predict_vertical_left(img)
            img = img[:, (self.size - left % self.size):]
        if top:
            for i in range(top//self.size+1):
                img = self.predict_horizontal_top(img)
            img = img[(self.size - top % self.size):, :]
        if bottom:
            for i in range(bottom//self.size+1):
                img = self.predict_horizontal_bottom(img)
            img = img[:-(self.size - bottom % self.size), :]
        return img

    def predict_vertical_right(self, img):
        img = np.array(img)
        len_last_piece = img.shape[0] % self.size
        input_val = []
        for step in range(0, img.shape[0]-self.size, self.size):
            input_val.append(np.split(img[step:step+self.size, -3*self.size:], 3, 1))
        input_val.append([img[i*self.size-1:(i+1)*self.size-1, -self.size:] for i in range(-3, 0)])
        input_val = np.array(input_val)
        predict = self.model.predict(input_val)
        predict = np.concatenate(predict, axis=0)
        predict = np.delete(predict, np.s_[-self.size:-len_last_piece], 0)
        return np.append(img, predict, axis=1)

    def predict_vertical_left(self, img):
        img = np.array(img)
        len_last_piece = img.shape[0] % self.size
        input_val = []
        for step in range(0, img.shape[0]-self.size, self.size):
            input_val.append(np.split(img[step:step+self.size, :3*self.size], 3, 1)[::-1])
        input_val.append([img[i*self.size:(i+1)*self.size, :self.size] for i in range(3)][::-1])
        input_val = np.array(input_val)
        predict = self.model.predict(input_val)
        predict = np.concatenate(predict, axis=0)
        predict = np.delete(predict, np.s_[-self.size:-len_last_piece], 0)
        return np.append(predict, img, axis=1)

    def predict_horizontal_top(self, img):
        img = np.array(img)
        len_last_piece = img.shape[1] % self.size
        input_val = []
        for step in range(0, img.shape[1]-self.size, self.size):
            input_val.append(np.split(img[:3*self.size, step:step+self.size], 3, 0)[::-1])
        input_val.append(np.split(img[:3*self.size, -self.size:], 3, 0)[::-1])
        input_val = np.array(input_val)
        predict = self.model.predict(input_val)
        predict = np.concatenate(predict, axis=1)
        predict = np.delete(predict, np.s_[-self.size:-len_last_piece], 1)
        return np.append(predict, img, axis=0)

    def predict_horizontal_bottom(self, img):
        img = np.array(img)
        len_last_piece = img.shape[1] % self.size
        input_val = []
        for step in range(0, img.shape[1]-self.size, self.size):
            input_val.append(np.split(img[-3*self.size:, step:step+self.size], 3, 0))
        input_val.append(np.split(img[-3*self.size:, -self.size:], 3, 0))
        input_val = np.array(input_val)
        predict = self.model.predict(input_val)
        predict = np.concatenate(predict, axis=1)
        predict = np.delete(predict, np.s_[-self.size:-len_last_piece], 1)
        return np.append(img, predict, axis=0)


if __name__ == '__main__':
    image = np.array(load_img('../../foto1.jpg'))/255
    print(image.shape)
    q = ImageResizer()
    image = q.predict(image, left=200, bottom=110)
    print(image.shape)
    plt.imshow(image)
    plt.show()
