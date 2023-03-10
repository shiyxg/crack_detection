from pylab import *
from PIL import Image
import os
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, TensorBoard
from keras.preprocessing.image import ImageDataGenerator

from tools import bianry, confusion_matrix, special_plot
from tools import get_label, get_all_images, special_print, dataset_plot, plot_samples
from datasets import DataSetBalanced
from graph import UNET, UNET_RESNET
from tools import half_size, half_size_reverse, save_label_image



def get_crater_datasets():
    a = DataSetBalanced()
    # labelled = ['./Images/Shi_raw/row13_column47.png','./Images/Shi_raw/row14_column18.png']
    labelled = get_all_images('./Images/Shi_raw', wanted='.png', friend='_label.png')
    # a.load_images_multi_cores(labelled)
    a.__load_images__(labelled)
    dataset = a.get_dataset(0, 40*len(labelled), 80*len(labelled), 0*len(labelled), size=[256,256])
    np.save('DataSet/trainset.npy', dataset)

    # a.show_samples_multi_cores()
    for i in a.labelled:
        i.plot_picked()
        print('task: '+i.image_name)




def train_UNet_enhanced():
    '''
    This function is build to train a UNET with residual and two branches based on ./DataSet/example.npy.
    I will use Image Preprocessing method in keras to do some enhancements.
    The loss function in keras is sigmoid, alpha=0.89,
    '''
    # network = UNET_RESNET(INPUT_SHAPE=[256, 256, 3], class_num=2)
    network = UNET(INPUT_SHAPE=[256, 256, 3], class_num=2)
    network.do_BN = True
    network.do_dropout = True
    network.do_transpose = False
    model = network.build_network(use_focal_loss=True, alpha=0.9)
    
    path = './logs/Crater_UNet2'

    callbacks = callbacks = [
        TensorBoard(log_dir=path + '/log', batch_size=1, histogram_freq=0, write_graph=True),
        ModelCheckpoint(filepath=path + '/model_check_point_latest', save_best_only=False, save_weights_only=True),
        ModelCheckpoint(filepath=path + '/model_check_point_best', save_best_only=True, save_weights_only=True),
        ReduceLROnPlateau(factor=0.3, patience=20, min_lr=0.00001)
    ]

    train_set = np.load('DataSet/trainset.npy', allow_pickle=True)
    print('train data load finished')
    x = train_set[:, :, :, 0:3]
    y = train_set[:, :, :, 3:]
    # get testset
    test_set = np.load('DataSet/trainset.npy', allow_pickle=True)
    print('test data load finished')
    x_test = test_set[:, :, :, 0:3]
    y_test = test_set[:, :, :, 3:]

    # data generator
    image_datagen_arg = dict(featurewise_center=True,
                             featurewise_std_normalization=True,
                             horizontal_flip=True,
                             vertical_flip=True,
                             rotation_range=45,
                             # width_shift_range=0.1,
                             # height_shift_range=0.1,
                             # shear_range=0.2,
                             zoom_range=0.2,
                             fill_mode='reflect',
                             cval=0
                         )
    mask_datagen_arg = dict( featurewise_center=True,
                             featurewise_std_normalization=True,
                             horizontal_flip=True,
                             vertical_flip=True,
                             rotation_range=45,
                             # width_shift_range=0.1,
                             # height_shift_range=0.1,
                             # shear_range=0.2,
                             zoom_range=0.2,
                             fill_mode='reflect',
                             cval=0,
                             preprocessing_function=bianry
                         )
    image_datagen = ImageDataGenerator(**image_datagen_arg)
    mask_datagen = ImageDataGenerator(**mask_datagen_arg)
    seed = 1
    batchsize=2
    image_generator = image_datagen.flow(x, seed=seed, batch_size=batchsize)
    mask_generator = mask_datagen.flow(y, seed=seed, batch_size=batchsize)
    train_generator = zip(image_generator, mask_generator)
    print(1111)
    model.summary()
    # special_plot(train_generator) # to show the result of train_generator
    # model.load_weights('logs/UNET_Dan_03/model_check_point_latest')
    history = model.fit_generator(train_generator, steps_per_epoch=200, epochs=100, callbacks=callbacks,
                                  validation_data=[x_test, y_test])
    model.save(path + '/model_parameter')
    
def predict(name):
    image_name = './Images/Shi_raw/{}.png'.format(name)
    image_raw = Image.open(image_name)
    image = np.array(image_raw)[..., 0:3].astype('float32')
    w_o,h_o = image_raw.size
    size = list(image_raw.size)
    if size[0] % 32 != 0:
        size[0] = (size[0] // 32 + 1) * 32
    if size[1] % 32 != 0:
        size[1] = (size[1] // 32 + 1) * 32
    image_raw = image_raw.resize(size)

    image = np.array(image_raw).astype('float32')[:,:,0:3]
    w, h, c = image.shape
    image = image.reshape([1, w, h, c])

    network = UNET_RESNET(INPUT_SHAPE=[w, h, c], class_num=2)
    # network = UNET(INPUT_SHAPE=[w, h, c], class_num=2)
    network.do_BN = True
    network.do_dropout = True
    network.do_transpose = False
    model = network.build_network(use_focal_loss=True)

    weight = 'logs/Crater_UNet1/model_check_point_best'
    model.load_weights(weight)

    image_bn = image.astype('float32')
    # image_bn = (image-128)/128
    s = time.time()

    # image_bn = half_size(image_bn)
    r = model.predict(image_bn, batch_size=1)
    # r = half_size_reverse(r)
    r = r[0,:,:,0]
    print('get a image prediction cast:{}s'.format(time.time() - s))
    file_name = image_name.replace('.png', '_'+weight.split('/')[1] + '.png')
    save_label_image(r.copy(), image, file_name, 0.3, size=[w_o,h_o], one_color=0)
    if 0:
        label = Image.open(image_name.replace('.png', '_label.png'))
        label = get_label(label)
        pred = np.zeros_like(r)
        thre = r.max()*0.6
        pred[where(r>thre)]=1
        TP, WP, WN, TN = confusion_matrix(label, pred)
        print('thre', thre)
        print('''{:<8}\t{:<8}\n{:<8}\t{:<8}'''.format(TP, WP, WN, TN))
        plot_label(r, image[0,:,:,:], human_label=None, thre=0.5, one_color=True)


if __name__ == "__main__":
    # get_crater_datasets()
    # train_UNet_enhanced()
    predict('row13_column47')

