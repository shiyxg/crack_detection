from pylab import *
from PIL import Image
import os
from tools import get_label, get_all_images, special_print, dataset_plot, plot_samples,skeleton_confusion_matrix

from datasets import DroneImage, DataSetBalanced

def drone_image_test():
    a = DroneImage('./Images/Shi_raw/row13_column47.png', load=True, label_appendix='_label.png', pred_appendix='_pred.png')

    size=[256, 256]
    for k in range(80):
        a.get_positive(size)
    for k in range(80):
        a.get_random(size)
    a.plot_picked()


def DataSet_Balanced_test():
    a = DataSetBalanced()
    labelled = get_all_images('./drones/old', wanted='.JPG')
    special = get_all_images('./drones/special_edges', wanted='.jpg')

    # for i in special:
    #     c = Image.open(i)
    #     if c.size[0] < 550 or c.size[1] < 550:
    #         special.remove(i)

    # labelled = labelled[-7:]
    # special = special[-7:]
    # a.load_images_multi_cores(labelled, special)
    # c = a.get_dataset(special_num=30*len(special), positive_num=100*len(labelled), noise_num=100*len(labelled), random_num=100*len(labelled), size=[256,256])
    # a.show_samples_multi_cores()
    # np.save('DataSet/Drones_balanced_test_fixed.npy', c)

    labelled = labelled[:-7]
    special = special[:-7]
    a.load_images_multi_cores(labelled, special)
    c = a.get_dataset(special_num=30 * len(special), positive_num=100 * len(labelled), noise_num=100 * len(labelled),
                      random_num=100 * len(labelled), size=[256, 256])
    np.save('DataSet/Drones_balanced_train_fixed.npy', c)
    a.show_samples_multi_cores()


def example():
    a = DataSetBalanced()
    labelled = ['./Images/Shi_raw/row13_column47.png','./Images/Shi_raw/row14_column18.png']
    # a.load_images_multi_cores(labelled)
    a.__load_images__(labelled)
    dataset = a.get_dataset(0, 50*len(labelled), 100*len(labelled), 0*len(labelled), size=[256,256])
    np.save('DataSet/example.npy', dataset)
    # a.show_samples_multi_cores()
    for i in a.labelled:
        i.plot_picked()
        print('task: '+i.image_name)

if __name__ == "__main__":
    # drone_image_test()
    # DataSet_Balanced_test()
    # DataSet_top()
    # DataSet_dan()
    # DataSet_top_new()
    # DataSet_correct()
    # DataSet_top_dan()
    example()
