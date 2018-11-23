from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
import warnings
import keras
import preprocessing_utils
import pretrained_model
import loss_utils
import optimizerhandler

#SET BACKEND CONFIGURATIONS
keras.backend.set_image_data_format('channels_last')
keras.backend.set_image_dim_ordering('tf')
#END SET BACKEND CONFIGURATIONS

#LOAD IMAGES
CONTENT_IMAGE_PATH = "/path/to/content/image"
STYLE_IMAGE_PATH   = "/path/to/content/image"
FINAL_IMAGE_PATH   = "/path/to/content/image"
PRETRAINED_MODEL   = "vgg16"
MIN_IMPROVEMENT_THRESHOLD = "0.00001"

content_image = keras.backend.variable(preprocessing_utils.preprocess_image(CONTENT_IMAGE_PATH))
style_reference_images = []
for style_path in STYLE_IMAGE_PATH:
    style_images.append(keras.backend.variable(preprocess_image(style_path)))

nb_style_images = len(style_images)
final_image = keras.backend.placeholder((1, img_width, img_height, 3))
#END LOADING IMAGES

#CREATE TENSORS
image_tensors = [content_image]
for style_image in style_images:
    image_tensors.append(style_image)
image_tensors.append(final_image)

nb_tensors = len(image_tensors)

# combine the various images into a single Keras tensor
input_tensor = keras.backend.concatenate(image_tensors, axis=0)
#END CREATE TENSORS

#LOAD MODEL
shape = (nb_tensors, img_width, img_height, 3)

print('Load model.')
ip = keras.layersInput(tensor=input_tensor, batch_shape=shape)
pretrained_model = pretrained_model.build(PRETRAINED_MODEL)
model = keras.model.Model(ip, pretrained_model)
print('Model loaded.')
#END LOAD MODEL

optimizerHandler = optimizerHandler(final_image, model,... )

# run scipy-based optimization (L-BFGS) over the pixels of the generated image
# so as to minimize the neural style loss
x = preprocessing_utils.preprocess_image(CONTENT_IMAGE_PATH, True, read_mode=read_mode)
num_iter = NUM_ITERATIONS
prev_min_val = -1
improvement_threshold = float(MIN_IMPROVEMENT_THRESHOLD)

for i in range(num_iter):
    print("Starting iteration %d of %d" % ((i + 1), num_iter))
    start_time = time.time()

    x, min_val, info = fmin_l_bfgs_b(optimizerHandler.loss, x.flatten(), fprime=optimizerHandler.grads, maxfun=20)

    if prev_min_val == -1:
        prev_min_val = min_val

    improvement = (prev_min_val - min_val) / prev_min_val * 100

    print('Current loss value:', min_val, " Improvement : %0.3f" % improvement, "%")
    prev_min_val = min_val
    # save current generated image
    img = deprocess_image(x.copy())

    if preserve_color and content is not None:
        img = original_color_transform(content, img, mask=color_mask)

    if not rescale_image:
        img_ht = int(img_width * aspect_ratio)
        print("Rescaling Image to (%d, %d)" % (img_width, img_ht))
        img = imresize(img, (img_width, img_ht), interp=args.rescale_method)

    if rescale_image:
        print("Rescaling Image to (%d, %d)" % (img_WIDTH, img_HEIGHT))
        img = imresize(img, (img_WIDTH, img_HEIGHT), interp=args.rescale_method)

    fname = result_prefix + '_at_iteration_%d.png' % (i + 1)
    imsave(fname, img)
    end_time = time.time()
    print('Image saved as', fname)
    print('Iteration %d completed in %ds' % (i + 1, end_time - start_time))

    if improvement_threshold is not 0.0:
        if improvement < improvement_threshold and improvement is not 0.0:
            print("Improvement (%f) is less than improvement threshold (%f). Early stopping script." % (
                improvement, improvement_threshold))
            exit()