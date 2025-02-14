import numpy as np
import matplotlib.pyplot as plt
from gan_for_numbers_with_libels import NumbersModel


model = NumbersModel.load_model("models/lable_gan_10_epochs_best")

labels = np.arange(1, 10).reshape((9, 1)).repeat(9, axis=1).T.flatten()
prediction_img, prediction_labels = model.predict(labels)

_, axs = plt.subplots(9, 9, figsize=(9, 10))
axs = axs.flatten()
for img, pr, label, ax in zip(prediction_img, prediction_labels, labels, axs):
    ax.imshow(img, cmap="gray")
    ax.axis("off")
    ax.set_title(f'{np.argmax(label)}, {pr[0]:.2f}')

plt.show()
