import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from keras import layers, optimizers, losses, models, datasets, random, ops, metrics, saving, callbacks, utils
import joblib


class NumbersModelCallback(callbacks.Callback):
    def __init__(self, path_img, noise_dim):
        super().__init__()
        self.path_img = path_img
        self.noise_dim = noise_dim

    
    def on_epoch_end(self, epoch, logs=None):
        noise = np.random.normal(0, 1, (20, self.noise_dim))
        labels = np.append(np.arange(10), np.arange(10), axis=0)
        labels = utils.to_categorical(labels, 10)
        prediction = (cond_gan.generator.predict([labels, noise]) + 1)/2
        fig, axs = plt.subplots(2, 10, figsize=(10, 3))
        axs = axs.flatten()
        for img, label, ax in zip(prediction, labels, axs):
            ax.imshow(img, cmap="gray")
            ax.axis("off")
            ax.set_title(f'{np.argmax(label)}')
        fig.savefig(os.path.join(self.path_img, f'e_{epoch}.jpg'))
        plt.close(fig)


class NumbersModel(models.Model):
    def __init__(self, discriminator, generator, noise_dim):
        super().__init__()
        self.generator = generator
        self.discriminator = discriminator
        self.noise_dim = noise_dim
        self.gen_loss_tracker = metrics.Mean(name="generator_loss")
        self.disc_loss_tracker = metrics.Mean(name="discriminator_loss")

    def compile(self, d_optimizer, g_optimizer, loss_fn):
        super().compile()
        self.d_optimizer = d_optimizer
        self.g_optimizer = g_optimizer
        self.loss_fn = loss_fn

    def train_step(self, train_data):
        labels_images, real_images = train_data
        labels_images = utils.to_categorical(labels_images, 10)
        batch_size = ops.shape(real_images)[0]

        random_noise_vectors = random.normal((batch_size, self.noise_dim))
        random_labels_vectors = random.randint((batch_size,), 0, 10)
        random_labels_vectors = utils.to_categorical(random_labels_vectors, 10)

        generated_images = self.generator([random_labels_vectors, random_noise_vectors])

        x_train_imag = ops.concatenate([real_images, generated_images], axis=0)
        x_train_labels = ops.concatenate([labels_images, random_labels_vectors], axis=0)
        y_train = ops.concatenate([np.zeros((batch_size, 1)), ops.ones((batch_size, 1))], axis=0)

        with tf.GradientTape() as tape:
            logits = self.discriminator([x_train_labels, x_train_imag], training=True)
            disc_loss_value = self.loss_fn(y_train, logits)
        grads_of_d = tape.gradient(disc_loss_value, self.discriminator.trainable_variables)
        self.d_optimizer.apply_gradients(zip(grads_of_d, self.discriminator.trainable_variables))

        random_noise_vectors = random.normal((2*batch_size, self.noise_dim))
        random_labels_vectors = random.randint((2*batch_size,), 0, 10)
        random_labels_vectors = utils.to_categorical(random_labels_vectors, 10)
        with tf.GradientTape() as tape:
            logits = self.generator([random_labels_vectors, random_noise_vectors], training=True)
            logits = self.discriminator([random_labels_vectors, logits], training=False)
            gen_loss_value = self.loss_fn(ops.zeros((2*batch_size, 1)), logits)
        gen_gradients = tape.gradient(gen_loss_value, self.generator.trainable_variables)
        self.g_optimizer.apply_gradients(zip(gen_gradients, self.generator.trainable_variables))

        self.gen_loss_tracker.update_state(gen_loss_value)
        self.disc_loss_tracker.update_state(disc_loss_value)
        return {
            "g_loss": self.gen_loss_tracker.result(),
            "d_loss": self.disc_loss_tracker.result(),
        }

    def predict(self, input_labels: np.array):
        labels = utils.to_categorical(input_labels, 10)
        noise = np.random.normal(0, 1, (input_labels.shape[0], self.noise_dim))
        predicted_img = (self.generator.predict([labels, noise]) + 1)/2
        predicted_labels = self.discriminator.predict([labels, predicted_img])
        return predicted_img, predicted_labels
    
    def summary(self, line_length=None, positions=None, print_fn=None, expand_nested=False, show_trainable=False, layer_range=None):
        return {
            "discriminator": self.discriminator.summary(line_length, positions, print_fn, expand_nested, show_trainable, layer_range),
            "generator": self.generator.summary(line_length, positions, print_fn, expand_nested, show_trainable, layer_range)
        }

    def save_model(self, path):
        if not os.path.exists(path):
            os.makedirs(path)
        self.generator.save(path + "/generator.keras")
        self.discriminator.save(path + "/discriminator.keras")
        joblib.dump({
                        "noise_dim": self.noise_dim,
                        "d_optimizer": self.d_optimizer,
                        "g_optimizer": self.g_optimizer,
                        "loss_fn": self.loss_fn
                    },
                    path + "/config.pkl")

    @classmethod
    def load_model(cls, path):
        discriminator: models.Model = saving.load_model(path + "/discriminator.keras")
        generator: models.Model = saving.load_model(path + "/generator.keras")
        config: dict = joblib.load(path + "/config.pkl")
        discriminator.name = "Discriminator"
        generator.name = "Generator"

        model = cls(discriminator, generator, config['noise_dim'])
        model.compile(config['d_optimizer'], config['g_optimizer'], config['loss_fn'])

        return model


if __name__ == "__main__":
    (y_train, x_train), _ = datasets.mnist.load_data()
    y_train = (y_train - 127.5) / 127.5
    y_train = y_train.reshape((-1, 28, 28, 1))

    batch_size = 32
    noise_dim = 100
    

    disc_input_labels = layers.Input(shape=(10,))
    disc_labels = layers.Dense(50)(disc_input_labels)
    disc_labels = layers.Dense(784)(disc_labels)
    disc_labels = layers.Reshape((28, 28, 1))(disc_labels)

    disc_input_image = layers.Input(shape=(28, 28, 1))

    disc = layers.Concatenate()([disc_input_image, disc_labels])
    disc = layers.Conv2D(128, (3,3), strides=(2, 2), padding='same')(disc)
    disc = layers.LeakyReLU(0.2)(disc)
    disc = layers.Flatten()(disc)
    disc = layers.Dropout(0.4)(disc)
    disc = layers.Dense(1, activation='sigmoid')(disc)

    discriminator = models.Model([disc_input_labels, disc_input_image], disc)
    
    gen_input_noise = layers.Input((noise_dim,))
    gen_noise = layers.Dense(32*7*7)(gen_input_noise)
    gen_noise = layers.LeakyReLU(negative_slope=0.2)(gen_noise)
    gen_noise = layers.Reshape((7, 7, 32))(gen_noise)

    gen_input_labels = layers.Input(shape=(10,))
    gen_labels = layers.Dense(50)(gen_input_labels)
    gen_labels = layers.Dense(49)(gen_labels)
    gen_labels = layers.Reshape((7, 7, 1))(gen_labels)

    gen = layers.Concatenate()([gen_noise, gen_labels])
    gen = layers.Conv2DTranspose(32, (4, 4), strides=(2, 2), padding="same")(gen)
    gen = layers.LeakyReLU(negative_slope=0.2)(gen)
    gen = layers.Conv2DTranspose(32, (4, 4), strides=(2, 2), padding="same")(gen)
    gen = layers.LeakyReLU(negative_slope=0.2)(gen)
    gen = layers.Conv2D(1, (7, 7), padding="same", activation="tanh")(gen)

    generator = models.Model([gen_input_labels, gen_input_noise], gen)

    cond_gan = NumbersModel(
        discriminator=discriminator, generator=generator, noise_dim=noise_dim
    )

    cond_gan.compile(
        d_optimizer=optimizers.Adam(learning_rate=0.0003, beta_1 = 0.5, beta_2=0.999),
        g_optimizer=optimizers.Adam(learning_rate=0.0003, beta_1 = 0.5, beta_2=0.999),
        loss_fn=losses.BinaryCrossentropy(),
    )
    try:
        cond_gan.fit(x_train, y_train, epochs=10, callbacks=[NumbersModelCallback('try_gen/training_fig',
                                                                         noise_dim)])
    except KeyboardInterrupt as e:
        pass

    cond_gan.save_model('models/k_gan')

    labels = np.arange(1, 10).reshape((9, 1)).repeat(9, axis=1).T.flatten()
    prediction_img, prediction_labels = cond_gan.predict(labels)

    _, axs = plt.subplots(9, 9, figsize=(9, 10))
    axs = axs.flatten()
    for img, pr, label, ax in zip(prediction_img, prediction_labels, labels, axs):
        ax.imshow(img, cmap="gray")
        ax.axis("off")
        ax.set_title(f'{np.argmax(label)}, {pr[0]:.2f}')

