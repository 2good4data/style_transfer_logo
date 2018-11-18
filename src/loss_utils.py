import keras

def gram_matrix(x):
    assert keras.backend.ndim(x) == 3
    features = keras.backend.batch_flatten(keras.backend.permute_dimensions(x, (2, 0, 1)))
    gram = keras.backend.dot(features, keras.backend.transpose(features))
    return gram

# the "style loss" is designed to maintain
# the style of the reference image in the generated image.
# It is based on the gram matrices (which capture style) of
# feature maps from the style reference image
# and from the generated image
def style_loss(style, final, mask_path=None, nb_channels=None):
    assert keras.backend.ndim(style) == 3
    assert keras.backend.ndim(final) == 3

    if content_mask_path is not None:
        content_mask = keras.backend.variable(load_mask(content_mask_path, nb_channels))
        combination = combination * keras.backend.stop_gradient(content_mask)
        del content_mask

    if mask_path is not None:
        style_mask = keras.backend.variable(load_mask(mask_path, nb_channels))
        style = style * keras.backend.stop_gradient(style_mask)
        if content_mask_path is None:
            combination = combination * keras.backend.stop_gradient(style_mask)
        del style_mask

    S = gram_matrix(style)
    C = gram_matrix(combination)
    channels = 3
    size = img_width * img_height
    return keras.backend.sum(keras.backend.square(S - C)) / (4. * (channels ** 2) * (size ** 2))

# an auxiliary loss function
# designed to maintain the "content" of the
# base image in the generated image
def content_loss(content, final):
    channel_dim = 0 
    channels = keras.backend.shape(base)[channel_dim]
    size = img_width * img_height

    if args.content_loss_type == 1:
        multiplier = 1. / (2. * (channels ** 0.5) * (size ** 0.5))
    elif args.content_loss_type == 2:
        multiplier = 1. / (channels * size)
    else:
        multiplier = 1.

    return multiplier * keras.backend.sum(keras.backend.square(combination - base))

# the 3rd loss function, total variation loss,
# designed to keep the generated image locally coherent
def total_variation_loss(x):
    assert keras.backend.ndim(x) == 4
    a = keras.backend.square(x[:, :img_width - 1, :img_height - 1, :] - x[:, 1:, :img_height - 1, :])
    b = keras.backend.square(x[:, :img_width - 1, :img_height - 1, :] - x[:, :img_width - 1, 1:, :])
    return keras.backend.sum(keras.backend.pow(a + b, 1.25))