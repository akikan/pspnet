from keras.layers import Activation, Input, add, Concatenate ,Lambda
from keras.layers import Conv2D, AveragePooling2D
from keras.layers.normalization import BatchNormalization
from keras.backend import tf as ktf
from keras import Model
from keras import layers
import keras
from ResNetModel import ResNet

#このクラスは借り物です
class Interp(layers.Layer):

    def __init__(self, new_size, **kwargs):
        self.new_size = new_size
        super(Interp, self).__init__(**kwargs)

    def build(self, input_shape):
        super(Interp, self).build(input_shape)

    def call(self, inputs, **kwargs):
        new_height, new_width = self.new_size
        resized = ktf.image.resize_images(inputs, [new_height, new_width],
                                          align_corners=True)
        return resized

    def compute_output_shape(self, input_shape):
        return tuple([None, self.new_size[0], self.new_size[1], input_shape[3]])

    def get_config(self):
        config = super(Interp, self).get_config()
        config['new_size'] = self.new_size
        return config

def pyramid_parts(input,shape):
    x=Conv2D(filters=512, kernel_size=(1,1),strides=(1,1))(input)
    resized=Interp(shape)(x)
    return resized

def pyramidModule(input):
    x1=AveragePooling2D(pool_size=(1,1))(input)
    x2=AveragePooling2D(pool_size=(2,2))(input)
    x3=AveragePooling2D(pool_size=(3,3))(input)
    x4=AveragePooling2D(pool_size=(6,6))(input)
    
    shape = [int(input.shape[1]),int(input.shape[2])]
    return  Concatenate()([
                          input,
                          pyramid_parts(x1,shape),
                          pyramid_parts(x2,shape),
                          pyramid_parts(x3,shape),
                          pyramid_parts(x4,shape)
                         ])

def pspnet(img_rows, img_cols, img_channels, x_train,output_num):
    resnet=ResNet(img_rows, img_cols, img_channels, x_train)
    x=pyramidModule(resnet)

    x=Conv2D(512, (3, 3), strides=(3, 3))(x)
    x=BatchNormalization()(x)
    x=Activation('relu')(x)

    x=Conv2D(output_num, (1, 1), strides=(1, 1))(x)
    x=Activation('softmax')(x)

    model=Model(inputs=input,outputs=x)
    return model