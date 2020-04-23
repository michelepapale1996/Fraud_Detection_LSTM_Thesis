# Fast Gradient Sign Method
import tensorflow as tf
import keras.backend as K

# Get current session (assuming tf backend)
sess = K.get_session()

def craft_sample(sample_to_modify, model, epochs=1, epsilon=0.01):
    # Initialize adversarial example with input image
    x_adv = sample_to_modify

    for i in range(epochs):
        target = [tf.zeros(1)]

        # Get the loss and gradient of the loss wrt the inputs
        loss = -1 * K.binary_crossentropy(target, model.output)
        grads = K.gradients(loss, model.input)

        # Get the sign of the gradient
        delta = K.sign(grads[0])

        # Perturb the sample
        x_adv = x_adv + epsilon * delta

        value = sess.run(x_adv, feed_dict={model.input: sample_to_modify})
        return value