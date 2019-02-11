import numpy as np
import matplotlib.pyplot as plt
from keras import backend as K
from keras import models
import cv2

from utils.training import *

##############################POST PROCESSING TOOLS #########################################

def plot_results(i, predictions_array, img, batches):
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(img.astype(np.uint8),  interpolation='nearest')
    predicted_label = np.argmax(predictions_array[i])
    if(predicted_label == 0 ):
        predicted_label='cats'
    else:
        predicted_label='dogs'
    color='blue'
    plt.xlabel("{} {:2.0f}% ({})".format(predicted_label,
                                100*np.max(predictions_array),
                                (batches.class_indices[predicted_label]), color=color))
    plt.show()

########################### MODEL TOOLS #####################################################




#################################### VISUALIZATION FUNCTIONS ###################################

def visualize_activations(model, layer_name, img, activation_index):
    layer_output = model.get_layer(layer_name).output
    print(model.get_layer(layer_name).output_shape)
    activation_model = models.Model(inputs=model.input, outputs=layer_output)
    activations = activation_model.predict(img)
    layer_activation = activations[0]
    print(layer_activation.shape)

    plt.matshow(layer_activation[:, :, activation_index], cmap='viridis')
    plt.show()


def deprocess_image(x):
    # normalize tensor: center on 0., ensure std is 0.1
    x -= x.mean()
    x /= (x.std() + 1e-5)
    x *= 0.1

    # clip to [0, 1]
    x += 0.5
    x = np.clip(x, 0, 1)

    # convert to RGB array
    x *= 255
    x = np.clip(x, 0, 255).astype('uint8')
    return x


def generate_pattern_filter(model, layer_name, filter_index, size=150):
    # Build a loss function that maximizes the activation
    # of the nth filter of the layer considered.
    layer_output = model.get_layer(layer_name).output
    loss = K.mean(layer_output[:, :, :, filter_index])

    # Compute the gradient of the input picture wrt this loss
    grads = K.gradients(loss, model.input)[0]

    # Normalization trick: we normalize the gradient
    grads /= (K.sqrt(K.mean(K.square(grads))) + 1e-5)

    # This function returns the loss and grads given the input picture
    iterate = K.function([model.input], [loss, grads])

    # We start from a gray image with some noise
    input_img_data = np.random.random((1, size, size, 3)) * 20 + 128.

    # Run gradient ascent for 40 steps
    step = 1.
    for i in range(20):
        loss_value, grads_value = iterate([input_img_data])
        input_img_data += grads_value * step

    img = input_img_data[0]
    return deprocess_image(img)


def visualize_filters(model, layer_names):
    for layer_name in layer_names:
        size = 64
        num_filters = 64  # model.get_layer(layer_name).output_shape[-1]
        old_num_rows = num_filters // 8
        remain = num_filters % old_num_rows

        num_rows = old_num_rows
        if remain > 0:
            num_rows = old_num_rows + 1
        # Display the results grid
        fig = plt.figure(figsize=(20, 20))
        for i in range(num_filters):  # iterate over the rows of our results grid
            # Generate the pattern for filter `i + (j * 8)` in `layer_name`
            filter_img = generate_pattern_filter(model, layer_name, i, size=size)
            ax = fig.add_subplot(num_rows, 8, i + 1)
            ax.imshow(filter_img)
    plt.show()

# expects preprocessed image
def visualize_CAM(model, last_conv_layer_name, img):
    preds = model.predict(img)

    #This is required for a VGG trained on imagenet
    #print('Predicted:', decode_predictions(preds, top=3)[0])

    predited_class = np.argmax(preds)

    class_output = model.output[:, predited_class]

    # the last convolutional layer in the model
    last_conv_layer = model.get_layer(last_conv_layer_name)

    # This is the gradient of the predicted class with regard to
    # the output feature map of last conv layer
    grads = K.gradients(class_output, last_conv_layer.output)[0]


    # mean intensity of the gradient over a specific feature map channel
    pooled_grads = K.mean(grads, axis=(0, 1, 2))

    # This function allows us to access the values of the quantities we just defined:
    # `pooled_grads` and the output feature map of last conv layer,
    # given a sample image
    iterate = K.function([model.input], [pooled_grads, last_conv_layer.output[0]])


    pooled_grads_value, conv_layer_output_value = iterate([img])


    for i in range(pooled_grads_value.shape[0]):
        conv_layer_output_value[:, :, i] *= pooled_grads_value[i]

    heatmap = np.mean(conv_layer_output_value, axis=-1)
    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap)

    # We resize the heatmap to have the same size as the original image
    heatmap = cv2.resize(heatmap, (img.shape[2], img.shape[1]))

    # We convert the heatmap to RGB
    heatmap = np.uint8(255 * heatmap)

    # We apply the heatmap to the original image
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    # 0.4 here is a heatmap intensity factor
    superimposed_img = heatmap * 0.4 + img[0]
    plt.imshow( deprocess_image(superimposed_img))
