import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from keras import layers, optimizers, losses, models, datasets, random, ops, metrics, saving, callbacks
import joblib


class NumbersModelCallback(callbacks.Callback):
    def __init__(self, path_img, latent_dim):
        super().__init__()
        self.path_img = path_img
        self.latent_dim = latent_dim


    def on_epoch_end(self, epoch, logs=None):
        noise = np.random.normal(0, 1, (5, self.latent_dim))
        prediction = generator.predict(noise)
        fig, axs = plt.subplots(1, 5)
        axs = axs.flatten()
        for img, ax in zip(prediction, axs):
            ax.imshow(img, cmap="gray")
        fig.savefig(os.path.join(self.path_img, f'e_{epoch}.jpg'))
        plt.close(fig)


class NumbersModel(models.Model):
    def __init__(self, discriminator, generator, latent_dim):
        super().__init__()
        self.generator = generator
        self.discriminator = discriminator
        self.latent_dim = latent_dim
        self.gen_loss_tracker = metrics.Mean(name="generator_loss")
        self.disc_loss_tracker = metrics.Mean(name="discriminator_loss")

    def compile(self, d_optimizer, g_optimizer, loss_fn):
        super().compile()
        self.d_optimizer = d_optimizer
        self.g_optimizer = g_optimizer
        self.loss_fn = loss_fn

    def train_step(self, real_images):
        batch_size = ops.shape(real_images)[0]

        random_latent_vectors = random.normal((batch_size, self.latent_dim))
        generated_images = self.generator(random_latent_vectors)
        x_train = ops.concatenate([real_images, generated_images], axis=0)
        y_train = ops.concatenate([ops.zeros((batch_size, 1)), ops.ones((batch_size, 1))], axis=0)

        with tf.GradientTape() as tape:
            logits = self.discriminator(x_train, training=True)
            disc_loss_value = self.loss_fn(y_train, logits)
        grads_of_d = tape.gradient(disc_loss_value, self.discriminator.trainable_variables)
        self.d_optimizer.apply_gradients(zip(grads_of_d, self.discriminator.trainable_variables))
        
        random_latent_vectors = random.normal((batch_size, self.latent_dim))
        with tf.GradientTape() as tape:
            logits = self.generator(random_latent_vectors, training=True)
            logits = self.discriminator(logits, training=False)
            gen_loss_value = self.loss_fn(ops.zeros((batch_size, 1)), logits)
        gen_gradients = tape.gradient(gen_loss_value, self.generator.trainable_variables)
        self.g_optimizer.apply_gradients(zip(gen_gradients, self.generator.trainable_variables))

        self.gen_loss_tracker.update_state(gen_loss_value)
        self.disc_loss_tracker.update_state(disc_loss_value)
        return {
            "g_loss": self.gen_loss_tracker.result(),
            "d_loss": self.disc_loss_tracker.result(),
        }
    
    def save_model(self, path):
        if not os.path.exists(path):
            os.makedirs(path)
        self.generator.save(path + "/generator.keras")
        self.discriminator.save(path + "/discriminator.keras")
        joblib.dump({
                        "latent_dim": self.latent_dim,
                        "d_optimizer": self.d_optimizer,
                        "g_optimizer": self.g_optimizer,
                        "loss_fn": self.loss_fn
                    },
                    path + "/config.pkl")

    @classmethod
    def load_model(cls, path):
        discriminator = saving.load_model(path + "/discriminator.keras")
        generator = saving.load_model(path + "/generator.keras")
        config = joblib.load(path + "/config.pkl")
        
        model = cls(discriminator, generator, config['latent_dim'])
        model.compile(config['d_optimizer'], config['g_optimizer'], config['loss_fn'])
        
        return model


if __name__ == "__main__":
    (x_train, _), _ = datasets.mnist.load_data()
    x_train = x_train/255
    x_train = x_train.reshape((-1, 28, 28, 1))

    batch_size = 32
    latent_dim = 128

    discriminator = models.Sequential(
        [
            layers.Input((28, 28, 1)),
            layers.Conv2D(64, (3, 3), strides=(2, 2), padding="same"),
            layers.LeakyReLU(negative_slope=0.2),
            layers.Conv2D(128, (3, 3), strides=(2, 2), padding="same"),
            layers.LeakyReLU(negative_slope=0.2),
            layers.GlobalMaxPooling2D(),
            layers.Dense(1, activation='sigmoid')
        ],
        name="discriminator",
    )

    generator = models.Sequential(
        [
            layers.Input((latent_dim,)),
            layers.Dense(7 * 7 * 10),
            layers.LeakyReLU(negative_slope=0.2),
            layers.Reshape((7, 7, 10)),
            layers.Conv2DTranspose(128, (4, 4), strides=(2, 2), padding="same"),
            layers.LeakyReLU(negative_slope=0.2),
            layers.Conv2DTranspose(128, (4, 4), strides=(2, 2), padding="same"),
            layers.LeakyReLU(negative_slope=0.2),
            layers.Conv2D(1, (7, 7), padding="same", activation="sigmoid"),
        ],
        name="generator",
    )

    cond_gan = NumbersModel(
        discriminator=discriminator, generator=generator, latent_dim=latent_dim
    )

    cond_gan.compile(
        d_optimizer=optimizers.Adam(learning_rate=0.0002, beta_1 = 0.5, beta_2=0.999),
        g_optimizer=optimizers.Adam(learning_rate=0.0002, beta_1 = 0.5, beta_2=0.999),
        loss_fn=losses.BinaryCrossentropy(),
    )

    try:
        cond_gan.fit(x_train[:320], epochs=10, callbacks=[NumbersModelCallback('try_gen/training_fig',
                                                                         latent_dim)])
    except KeyboardInterrupt as e:
        pass

    cond_gan.save_model('models/k_gan')


    noise = np.random.normal(0, 1, (9, latent_dim))
    prediction = cond_gan.generator.predict(noise)

    _, axs = plt.subplots(3, 3, figsize=(12, 12))
    axs = axs.flatten()
    for img, ax in zip(prediction, axs):
        ax.imshow(img, cmap="gray")
    plt.show()

