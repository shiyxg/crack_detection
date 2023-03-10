from pylab import *
from PIL import Image
import os
from tools import get_label, get_all_images, special_print, dataset_plot, plot_samples,skeleton_confusion_matrix
from tools import Gaussian_blur2d_faster as blur
from tools import get_image_skeleton, save_label_image
from tools import skeleton_confusion_matrix, confusion_matrix_print
from concurrent.futures import ProcessPoolExecutor


class DroneImage:
    def __init__(self, image_name, label_name=None, pred_name=None, load=True, label_format='jpg', **keys):
        if keys.get('image_appendix') is None:
            image_appendix='.png'
        else:
            image_appendix  = keys['image_appendix']
        if label_name is None:
            if keys.get('label_appendix') is None:
                label_name = image_name.replace(image_appendix, '_label.png')
            else:
                label_name = image_name.replace(image_appendix, keys['label_appendix'])
        if pred_name is None:
            if keys.get('pred_appendix') is None:
                pred_name = image_name.replace(image_appendix, '_pred.png')
            else:
                pred_name = image_name.replace(image_appendix, keys['pred_appendix'])

        self.image_name = image_name
        self.label_name = label_name
        self.pred_name = pred_name

        self.size = []
        self.crack_num = 0
        self.noise_num = 0
        self.pick_index = []
        self.too_close=False
        self.close_standard = 0
        self.close_standard_positive = 70
        self.close_standard_noise = 100
        self.close_standard_random = 150
        if load:
            self.__load__()

    def __load__(self):
        self.raw_image = Image.open(self.image_name)
        self.image = np.array(self.raw_image)[...,0:3]
        try:
            self.raw_label = Image.open(self.label_name)
            self.label = get_label(self.raw_label)
        except IOError:
            print(self.image_name, 'can Not read label, replace with zeros')
            self.raw_label = []
            self.label = np.zeros(self.image.shape[0:2])
        try:
            self.raw_pred = Image.open(self.pred_name)
            self.pred = get_label(self.raw_pred)
        except IOError:
            print(self.image_name, 'can Not read pred, replace with zeros')
            self.raw_pred = []
            self.pred = np.zeros(self.image.shape[0:2])

        self.crack_index = where(self.label == 1)

        skeleton_pred = self.pred

        self.noise_index = where((skeleton_pred - self.label) == 1)

        self.size = self.label.shape
        self.crack_num = len(self.crack_index[0])
        self.noise_num = len(self.noise_index[0])

    def get_subimages(self, index, size, kind=0):
        # return a 256*256 sample
        x,y = index
        w,h = self.size
        a,b=x,y
        x = min(max(x, size[0]//2), w-size[0]//2)
        y = min(max(y, size[1]//2), h-size[1]//2)
        # print(a,b,x, y)
        self.too_close=False
        for point in self.pick_index:
            p_x,p_y = (point[0][0]+point[0][2])/2, (point[1][0]+point[1][2])/2
            if abs(p_x-x)+abs(p_y-y)<self.close_standard:
                self.too_close=True
                break
        self.pick_index.append([[x-size[0]//2, x-size[0]//2, x+size[0]//2, x+size[0]//2, x-size[0]//2],
                                [y-size[1]//2, y+size[1]//2, y+size[1]//2, y-size[1]//2, y-size[1]//2],
                                kind])

        sample_image = self.image[(x-size[0]//2):(x+size[0]//2), (y-size[1]//2):(y+size[1]//2), :]
        sample_label = self.label[(x-size[0]//2):(x+size[0]//2), (y-size[1]//2):(y+size[1]//2)]
        return sample_image, sample_label

    def get_positive(self, size, shift=None):
        times = 0
        self.close_standard = self.close_standard_positive
        while 1:
            times = times+1
            if shift is None:
                shift = np.random.randint(-size[0]//4,size[0]//4, size=2)
            index = int(np.random.random()*self.crack_num)
            x= self.crack_index[0][index] + shift[0]
            y= self.crack_index[1][index] + shift[1]
            sample_image, sample_label  =  self.get_subimages([x,y], size, kind=1)

            if self.too_close and times < 10:
                self.pick_index.pop()
                continue
            else:
                break
        # print(times)
        return sample_image, sample_label

    def get_noise(self, size, shift=None):
        times = 0
        self.close_standard = self.close_standard_noise
        while 1:
            times = times + 1
            if shift is None:
                shift = np.random.randint(-size[0]//2,size[0]//2, size=2)
            index = int(np.random.random() * self.noise_num)
            x = self.noise_index[0][index] + shift[0]
            y = self.noise_index[1][index] + shift[1]
            sample_image, sample_label = self.get_subimages([x, y], size, kind=2)

            if self.too_close and times < 10:
                self.pick_index.pop()
                continue
            else:
                break
        # print(times)
        return sample_image, sample_label

    def get_random(self, size):
        # print(size[0]//2, self.size[0]-size[0]//2)
        times = 0
        self.close_standard = self.close_standard_random
        while 1:
            times = times + 1
            x = np.random.randint(size[0] // 2, self.size[0] - size[0] // 2)
            y = np.random.randint(size[1] // 2, self.size[1] - size[1] // 2)
            sample_image, sample_label = self.get_subimages([x, y], size, kind=3)

            if self.too_close and times < 10:
                self.pick_index.pop()
                continue
            else:
                break
        # print(times)
        return sample_image, sample_label

    def plot_picked(self):
        name = self.image_name.replace('.png', '_samples.png')
        name = name.replace('row','picked_row')
        plot_samples(self.image, self.label, self.pred, self.pick_index,
                     image_name=name)

    def confusion_matrix_length(self, tolerance=40, is_show=False, show_bg=False):
        if show_bg:
            TP, FP, FN, TN= skeleton_confusion_matrix(self.label, self.pred, bear_range=tolerance, is_show = is_show,
                                                      bg=self.image, title=self.image_name)
        else:
            TP, FP, FN, TN = skeleton_confusion_matrix(self.label, self.pred, bear_range=tolerance, is_show=is_show,
                                                       title=self.image_name)
        return TP, FP, FN, TN


class DataSetBalanced:
    '''
     This class will increase negative part in our dataset
    '''
    def __init__(self):

        self.labelled = []
        self.special = []
        self.weights = np.array([])

        self.special_num=len(self.special)
        self.labelled_num=len(self.labelled)
        self.label_appendix = '_label.png'
        self.pred_appendix = '_pred.jpg'
        self.image_appendix='png'
        self.close_standards=[70,100,150]

    def __load_images__(self, labelled, special=[]):
        s = time.time()
        for i in range(len(labelled)):
            image = self.__load_one_image__(labelled[i])
            self.labelled.append(image)
            special_print(labelled[i],len(labelled), i, s)

        s = time.time()
        for i in range(len(special)):
            image = DroneImage(special[i])
            self.special.append(image)
            special_print(special[i], len(special), i, s)

        self.special_num=len(self.special)
        self.labelled_num=len(self.labelled)

        weights = []
        for i in range(self.labelled_num):
            weights.append(self.labelled[i].crack_num)
        weights = np.array(weights)
        self.weights = weights/weights.sum()
        print('positive weight was set as :', self.weights)

    def __load_one_image__(self, name):
        result = DroneImage(name, load=True, label_appendix=self.label_appendix, pred_appendix=self.pred_appendix, image_appendix=self.image_appendix)
        [result.close_standard_positive,
         result.close_standard_noise,
         result.close_standard_random] = self.close_standards
        return result

    def load_images_multi_cores(self, labelled, special=[], cores_num=20):

        labelled_task = []
        special_task = []
        with ProcessPoolExecutor(cores_num) as executor:
            for i in labelled:
                task = executor.submit(self.__load_one_image__, **{'name':i})
                labelled_task.append(task)
                print('submit task: '+i)
            for i in special:
                task = executor.submit(self.__load_one_image__, **{'name': i})
                special_task.append(task)
                print('submit task: ' + i)
        print('all_tasks: finished')
        print('&'*50)
        for task in labelled_task:
            self.labelled.append(task.result())
        for task in special_task:
            self.special.append(task.result())

        self.special_num = len(self.special)
        self.labelled_num = len(self.labelled)

        weights = []
        for i in range(self.labelled_num):
            weights.append(self.labelled[i].crack_num)
        weights = np.array(weights)
        self.weights = weights/weights.sum()
        print('positive weight was set as :', self.weights)

    def show_samples_multi_cores(self, cores_num=20):

        with ProcessPoolExecutor(cores_num) as executor:
            for i in self.labelled:
                executor.submit(i.plot_picked)
                print('submit task: '+i.image_name)
            for i in self.special:
                executor.submit(i.plot_picked)
                print('submit task: ' + i.image_name)

        print('all_tasks: finished')

    def get_weighted_indices(self, start, end, num, p):
        # print(start, end, num, p)
        return np.random.choice(np.arange(start=start, stop=end), size=num,  p=p)

    def get_special(self, num, size):
        samples = np.zeros([num, size[0], size[1], 4])
        image_indices = np.random.randint(0, self.special_num, size=num)
        for i in range(num):
            image = self.special[image_indices[i]]
            sample, label = image.get_random(size)
            samples[i,:,:,0:3] = sample
            samples[i,:,:,3] = label
        return samples

    def get_positive(self, num, size):
        samples = np.zeros([num, size[0], size[1], 4])
        # image_indices = np.random.randint(0, self.labelled_num, size=num)
        image_indices = self.get_weighted_indices(0, self.labelled_num, num, self.weights)
        for i in range(num):
            image = self.labelled[image_indices[i]]
            sample, label = image.get_positive(size)
            samples[i, :, :, 0:3] = sample
            samples[i, :, :, 3] = label
        return samples

    def get_random(self, num, size):
        samples = np.zeros([num, size[0], size[1], 4])
        image_indices = np.random.randint(0, self.labelled_num, size=num)
        for i in range(num):
            image = self.labelled[image_indices[i]]
            sample, label = image.get_random(size)
            samples[i, :, :, 0:3] = sample
            samples[i, :, :, 3] = label
        return samples

    def get_noise(self, num, size):
        samples = np.zeros([num, size[0], size[1], 4])
        image_indices = np.random.randint(0, self.labelled_num, size=num)
        for i in range(num):
            image = self.labelled[image_indices[i]]
            sample, label = image.get_noise(size)
            samples[i, :, :, 0:3] = sample
            samples[i, :, :, 3] = label
        return samples

    def get_dataset(self, special_num=0, positive_num=0, random_num=0, noise_num=0, size=[256,256]):

        samples = []
        if special_num:
            samples.append(self.get_special(special_num, size).astype('uint8'))
        if positive_num:
            samples.append(self.get_positive(positive_num, size).astype('uint8'))
            print('positive:', positive_num, 'finished')
        if random_num:
            samples.append(self.get_random(random_num, size).astype('uint8'))
            print('random:', random_num, 'finished')
        if noise_num:
            samples.append(self.get_noise(noise_num, size).astype('uint8'))
            print('noise:', noise_num, 'finished')
        dataset = np.concatenate(samples, axis=0).astype('uint8')
        return dataset
