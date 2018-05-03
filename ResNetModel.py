from keras.layers import Activation, Input, add
from keras.layers import Conv2D
from keras.layers.normalization import BatchNormalization

def rescell(data, filters, kernel_size, option=False, dilated=(1,1)):
	strides=(1,1)
	if option:
		strides=(2,2)
	x=Conv2D(filters=filters, kernel_size=(1,1), dilation_rate=dilated, strides=strides, padding="same")(data)
	x=BatchNormalization()(x)
	x=Activation('relu')(x)

	x=Conv2D(filters=filters, kernel_size=(3,3), dilation_rate=dilated, strides=(1,1), padding="same")(x)
	x=BatchNormalization()(x)
	x=Activation('relu')(x)

	data=Conv2D(filters=filters*4, kernel_size=(1,1), strides=strides, padding="same")(data)
	x=Conv2D(filters=filters*4,kernel_size=(1,1),dilation_rate=dilated,strides=(1,1),padding="same")(x)
	x=BatchNormalization()(x)
	x=add([x,data])
	x=Activation('relu')(x)
	return x



def ResNet(img_rows, img_cols, img_channels, x_train):
	input=Input(shape=(img_rows,img_cols,img_channels))
	x=Conv2D(32,(2,2), padding="same", input_shape=x_train.shape[1:],activation="relu")(input)

	x=rescell(x,64,(3,3))
	x=rescell(x,64,(3,3))
	x=rescell(x,64,(3,3))

	x=rescell(x,128,(3,3),True)
	x=rescell(x,128,(3,3))
	x=rescell(x,128,(3,3))
	x=rescell(x,128,(3,3))

	x=rescell(x,256,(3,3),True)
	x=rescell(x,256,(3,3),dilated=(2,2))
	x=rescell(x,256,(3,3),dilated=(2,2))
	x=rescell(x,256,(3,3),dilated=(2,2))
	x=rescell(x,256,(3,3),dilated=(2,2))
	x=rescell(x,256,(3,3),dilated=(2,2))

	x=rescell(x,512,(3,3),True)
	x=rescell(x,512,(3,3),dilated=(4,4))
	x=rescell(x,512,(3,3),dilated=(4,4))
	return x