import keras

class Pretrained_model :
    @staticmethod
    def build(model_name):
        switch (model_name) {
            case "vgg16":
                model = keras.applications.vgg16.VGG16(weights='imagenet', include_top = False)
                break
            case "vgg19": 
                model = keras.applications.vgg19.VGG19(weights='imagenet', include_top = False)
                break
            default: 
                break
        }
        return model
