import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.api import layers, optimizers, losses, models, random, ops, metrics, saving, callbacks, utils
import joblib
from data_generator import SheetImageGenerator


class NumbersModelCallback(callbacks.Callback):
    def __init__(self, path_img, noise_dim, piece_size=(50, 50)):
        super().__init__()
        self.path_img = path_img
        if not os.path.exists(self.path_img):
            os.makedirs(self.path_img)
        self.noise_dim = noise_dim
        self.data_get = SheetImageGenerator('data/data2', 15, piece_size)
        
    def on_epoch_end(self, epoch, logs=None):
        noise = np.random.normal(0, 1, (5, self.noise_dim))
        (inp_images, _), (real_outputs, _) = self.data_get[np.random.randint(len(self.data_get))]
        prediction = imag_gan.generator.predict([inp_images, noise])
        fig, axs = plt.subplots(5, 3)
        for inp, r_out, pred, ax in zip(inp_images, real_outputs, prediction, axs):
            ax[0].imshow(np.concatenate(inp, axis=1))
            ax[1].imshow(r_out)
            ax[2].imshow(pred)
        fig.savefig(os.path.join(self.path_img, f'e_{epoch}.jpg'))
        plt.close(fig)


class ImagePiceGeneratorModel(models.Model):
    def __init__(self, discriminator: models.Model, generator:models.Model,
                 noise_dim=100):
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
        (input_images_dis, input_images_gen), output_image_dis = train_data
        batch_size_dis = ops.shape(input_images_dis)[0]
        batch_size_gen = ops.shape(input_images_gen)[0]

        random_noise_vectors = random.normal((batch_size_dis, self.noise_dim))
        generated_images = self.generator([input_images_dis, random_noise_vectors])

        x_input_imag = ops.concatenate([input_images_dis, input_images_dis], axis=0)
        x_output_imag = ops.concatenate([output_image_dis, generated_images], axis=0)
        y_train = ops.concatenate([ops.zeros((batch_size_dis, 1)), ops.ones((batch_size_dis, 1))], axis=0)

        with tf.GradientTape() as tape:
            logits = self.discriminator([x_input_imag, x_output_imag], training=True)
            disc_loss_value = self.loss_fn(y_train, logits)
        grads_of_d = tape.gradient(disc_loss_value, self.discriminator.trainable_variables)
        self.d_optimizer.apply_gradients(zip(grads_of_d, self.discriminator.trainable_variables))

        random_noise_vectors = random.normal((batch_size_gen, self.noise_dim))

        with tf.GradientTape() as tape:
            logits = self.generator([input_images_gen, random_noise_vectors], training=True)
            logits = self.discriminator([input_images_gen, logits], training=False)
            gen_loss_value = self.loss_fn(ops.zeros((batch_size_gen, 1)), logits)
        gen_gradients = tape.gradient(gen_loss_value, self.generator.trainable_variables)
        self.g_optimizer.apply_gradients(zip(gen_gradients, self.generator.trainable_variables))

        self.gen_loss_tracker.update_state(gen_loss_value)
        self.disc_loss_tracker.update_state(disc_loss_value)
        return {
            "g_loss": self.gen_loss_tracker.result(),
            "d_loss": self.disc_loss_tracker.result(),
        }

    def summary(self, line_length=None, positions=None, print_fn=None,
                expand_nested=False, show_trainable=False, layer_range=None):
        return {
            "discriminator": self.discriminator.summary(
                line_length, positions, print_fn, expand_nested, show_trainable, layer_range
                ),
            "generator": self.generator.summary(
                line_length, positions, print_fn, expand_nested, show_trainable, layer_range
                )
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


def build_generator(noise_dim, piece_size=(50, 50)):
    pass

def build_discriminator(piece_size=(50, 50)):
    pass


if __name__ == "__main__":
    batch_size = 33
    noise_dim = 100
    
    discriminator = build_discriminator()

    generator = build_generator(noise_dim)

    imag_gan = ImagePiceGeneratorModel(
        discriminator=discriminator, generator=generator, noise_dim=noise_dim
    )

    imag_gan.compile(
        d_optimizer=optimizers.Adam(learning_rate=0.0003, beta_1 = 0.5, beta_2=0.999),
        g_optimizer=optimizers.Adam(learning_rate=0.0003, beta_1 = 0.5, beta_2=0.999),
        loss_fn=losses.BinaryCrossentropy(),
    )

    imag_gan.summary()

    data_gen = SheetImageGenerator('data/data2', batch_size)
    callbacks=[NumbersModelCallback('traing/first_linear_image_generator', noise_dim)]

    try:
        imag_gan.fit(data_gen, epochs=5, callbacks=callbacks, steps_per_epoch=1000)
    except KeyboardInterrupt as e:
        pass

    imag_gan.save_model('models/first_square_image_generator')

