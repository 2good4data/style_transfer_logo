import keras
import loss_utils

class OptimizeHandler:
    def __init__(self, final_image, content_extraction_layers, style_extraction_layers):
        self.loss_value = None
        self.grads_values = None

        # get the symbolic outputs of each "key" layer (we gave them unique names).
		outputs_dict = dict([(layer.name, layer.output) for layer in model.layers])
		shape_dict   = dict([(layer.name, layer.output_shape) for layer in model.layers])
        # combine these loss functions into a single scalar
		loss = keras.backend.variable(0.)
		layer_features = outputs_dict[content_extraction_layers]  # 'conv5_2' or 'conv4_2'
		content_image_features = layer_features[0, :, :, :]
		final_features   = layer_features[nb_tensors - 1, :, :, :]
		loss = loss + content_weight * loss_utils.content_loss(base_image_features,
                                      combination_features)

		channel_index = -1

		feature_layers = ['conv1_1', 'conv2_1', 'conv3_1', 'conv4_1', 'conv5_1']
		for style_layer_name in style_extraction_layers:
    		
    		style_layer_features = outputs_dict[style_layer_name]
    		layer_shape = shape_dict[style_layer_name]
    		
    		final_features = layer_features[nb_tensors - 1, :, :, :]
    		style_reference_features = layer_features[1:nb_tensors - 1, :, :, :]

    		sl = []
    		for j in range(nb_style_images):
        		sl.append(loss_utils.style_loss(style_reference_features[j], final_features, style_masks[j], layer_shape))

    		for j in range(nb_style_images):
        		loss = loss + (style_weights[j] / len(feature_layers)) * sl[j]

			loss = loss + total_variation_weight * loss_utils.total_variation_loss(final_image)

			# get the gradients of the generated image wrt the loss
			grads = keras.backend.gradients(loss, final_image)

			outputs = [loss]
		if type(grads) in {list, tuple}:
   			outputs += grads
		else:
    		outputs.append(grads)

		self.graph_functions = keras.backend.function([final_image], outputs)

    def loss(self, x):
        assert self.loss_value is None
        Sx = x.reshape((1, img_width, img_height, 3))
    	outs = self.graph_functions([x])
    	loss_value = outs[0]
    	
    	if len(outs[1:]) == 1:
        	grad_values = outs[1].flatten().astype('float64')
    	else:
        	grad_values = np.array(outs[1:]).flatten().astype('float64')

        self.loss_value = loss_value
        self.grad_values = grad_values
        return self.loss_value

    def grads(self, x):
        assert self.loss_value is not None
        grad_values = np.copy(self.grad_values)
        self.loss_value = None
        self.grad_values = None
        return grad_values
