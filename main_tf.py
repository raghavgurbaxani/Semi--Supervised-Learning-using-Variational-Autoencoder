import input_data
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from models.variational_autoencoder import VariationalAutoencoder


def train(model, mnist_dataset, learning_rate=0.00005, batch_size=16,
          num_steps=5000):
    for step in range(0, num_steps):
        batch_x, _ = mnist_dataset.train.next_batch(batch_size)
        model.session.run(
                model.update_op_tensor,
                feed_dict={model.x_placeholder: batch_x,
                           model.learning_rate_placeholder: learning_rate}
                )


def main(_):
    # Get dataset.
    mnist_dataset = input_data.read_data_sets('MNIST_data', one_hot=True)

    # Build model.
    model = VariationalAutoencoder()

    # Start training
    train(model, mnist_dataset)

    # Plot out latent space, for +/- 3 std. #how is this latent space made and there is one latent space inside too.
    std = 1
    x_z = np.linspace(-3*std, 3*std, 20)
    y_z = np.linspace(-3*std, 3*std, 20) #linspace?

    out = np.empty((28*20, 28*20)) #Ask moitreya why 20? What is happening below?
    for x_idx, x in enumerate(x_z):
        for y_idx, y in enumerate(y_z):
            z_mu = np.array([[y, x]])
            img = model.generate_samples(z_mu)
            out[x_idx*28:(x_idx+1)*28,
                y_idx*28:(y_idx+1)*28] = img[0].reshape(28, 28)
    plt.imsave('latent_space.png', out, cmap="gray")

if __name__ == "__main__":
    tf.app.run()
