from pylab import *
from scipy import signal
import os
from skimage.filters import gaussian
from skimage.morphology import skeletonize, medial_axis
from PIL import Image
from concurrent.futures import ProcessPoolExecutor


def Gaussian_blur2d(var, target):
    '''
    generate a convolition of target and a Gaussian kernel
     var: size
    '''
    var_blur_x, var_blur_y = var
    x = np.linspace(-20, 20,41)
    y = np.linspace(-20, 20,41)
    xy, yx = np.meshgrid(x,y)
    xy = xy.T
    yx = yx.T
    Gaussian_blur = 0.4/var_blur_x * np.exp(-xy ** 2 / (2 * var_blur_x ** 2))*0.4/var_blur_y * np.exp(-yx ** 2 / (2 * var_blur_y ** 2))
    kernel = Gaussian_blur
    kernel = kernel / kernel.max()

    return signal.convolve2d(target, kernel,  mode='same')


def Gaussian_blur2d_faster(var, target):
    '''
    fast convolition of target and a Gaussian kernel
        var: valid size of kernel
        target: 2d matrix
    '''

    return gaussian(target, [var[0]/10, var[0]/10],  mode='reflect')


def mean_blur2d(var, target=None):
    '''
    blur target by mean, w,h = var is the size og blur filter
    '''
    var_x, var_y= var
    kernel = np.ones([var_x, var_y])/(var_x*var_y)
    return signal.convolve2d(target, kernel, mode='same')


def get_label(image, size = None, rlim=None, glim=None, blim=None):
    '''
    get label from a PIL.Image object. where 210<r,g<50,b<40 will be consider as label

        image: PIL.Image object
        size: whether resize image to size
        rlim: limit of r chns,  None or [low,high](rlim)
        glim, blim: value of highest level of a label pixel
    '''
    if size is not None:
        image = image.resize(size)
    label = np.array(image).astype('float32')
    r = label[:, :, 0]
    g = label[:, :, 1]
    b = label[:, :, 2]

    rlim = [210.8, 256.1] if rlim is None else rlim
    glim = 50.1 if glim is None else glim
    blim = 40.1 if blim is None else blim

    r = (np.sign(np.sign(r - rlim[0]) + np.sign(rlim[1] - r) - 1.1) + 1) / 2
    g = (np.sign(glim - g) + 1) / 2
    b = (np.sign(blim - b) + 1) / 2
    label = r * g * b
    return label


def save_label_image(label, image, file_name, thre=0.5, size=None, one_color=False):
    '''
    save label as image by PIL save.
        label: label, 2D array, from 0~1
        image: 3d array, w,h,c = image.shape
        file_name: file name to store
        thre: threshold for label
        size: whether to resize final image, [h, w] = size

    '''
    x, y = where(label > thre)

    if one_color:
        label[x,y]=1
    
    label[0, 0] = 0
    label[0, 1] = 1
    label_rgb = cm.bwr(label)[:, :, 0:3] * 255
    image_rgb = np.squeeze(image)
    image_rgb[x, y, :] = label_rgb[x, y, :]
    image_file = Image.fromarray(image_rgb.astype('uint8'), 'RGB')
    if size is not None:
        image_file = image_file.resize(size)
    image_file.save(file_name, quality=95)
    return None


def save_label_npy(label, file_name):
    '''
    save label as npy file as uint8.
        label: label, 2D array, from 0~1
        file_name: file name to store
    '''
    label = (label*255).astype('uint8')
    np.save(file_name, label)
    return None


class SaveLabel():
    '''
    save label class, use mutiprocesses to save time.
    save_raw: whether save npy file or not
    save_image: save image or not
    thre: 
    appendixJPG: for image, what appendix
    appendixNPY: for npy file, what appendix
    works_num: processes number
    self.image_dir, self.label_dir, self.npy_dir = dirs, used to replace raw_file_name
    '''

    def __init__(self, save_raw=True, save_image=True,
                 thre = 0.5, appendixJPG = '.JPG', appendixNPY = '.npy',
                 works_num = 16,
                 dirs = ['', '', ''], image_append='.JPG'):
        self.save_raw = save_raw
        self.save_image = save_image
        if not save_raw and not save_image:
            raise ValueError('nothing saved')
        self.thre = thre
        self.size = size
        self.appendix1 = appendixJPG
        self.appendix2 = appendixNPY
        self.exe = ProcessPoolExecutor(works_num)
        self.tasks = []
        self.tasks_num = 0
        self.image_append = image_append

        self.image_dir, self.label_dir, self.npy_dir  = dirs

    def submit_job(self, label, image, raw_file_name, size = None, thre = None):

        if self.label_dir is not '':
            file_name_label = raw_file_name.replace(self.image_dir, self.label_dir)
        else:
            file_name_label = raw_file_name
        file_name_label = file_name_label.replace(self.image_append, self.appendix1)

        if self.npy_dir is not '':
            file_name_npy = raw_file_name.replace(self.image_dir, self.npy_dir)
        else:
            file_name_npy = raw_file_name
        file_name_npy = file_name_npy.replace(self.image_append, self.appendix2)

        if self.save_image:
            if thre is None:
                thre = self.thre
            task = self.exe.submit(save_label_image, **{'label':label,
                                                       'image':image,
                                                       'file_name':file_name_label,
                                                       'size':size,
                                                       'thre':thre,})
            #task = save_label_image(**{'label':label,
            #                           'image':image,
            #                           'file_name':file_name_label,
            #                           'size':size,
            #                           'thre':thre,})
            self.tasks.append(task)
            self.tasks_num = self.tasks_num+1
        if self.save_raw:
            save_label_npy(label, file_name_npy)

    def wait(self):
        while 1:
            s = time.time()
            num = 0
            for task in self.tasks:
                if not task.done():
                    num = num+1
            if num==0:
                break
            pause(0.5)
            special_print(num, self.tasks_num, self.tasks_num-num+1, s)


def plot_label(label, image, human_label=None, thre = 0.5, title_name=None, one_color=False):
    '''
    plot label by plt.
        label: label, 2D array, from 0~1
        image: 3d array, w,h,c = image.shape
        thre: threshold for label
        one_color: plot label binaryly(red and Nothing)
    '''
    figure(figsize=[6,10], dpi=200)
    if one_color:
        label[where(label>thre)]= 1
    label[where(label<=thre)]= None
    label[0,0:2]=[0,1]
    imshow(image/256)
    imshow(label, cmap='bwr')
    if human_label is not None:
        c = np.ones_like(human_label)*1.0
        c[where(human_label==0)]=None
        c[0,0:2]=[0,1]
        imshow(c, cmap='gray_r')
    if title_name is not None:
        title(title_name)
    show()


def get_all_images(image_dir, wanted='.png', friend=None):
    '''
    get all wanted files from image_dir, 
        image_dir: 
        wanted: appendix of wanted files
        friend: replace wanted with friend, when new file exist, add this file
    
    return: a matched file names list
    '''
    images = []
    for dirpath, dirnames, filenames in os.walk(image_dir):
        for file in filenames:
            if 'picked' in file:
                continue
            if wanted in file:
                if friend is not None:
                    friend_name = file.replace(wanted,friend)

                    if friend_name in filenames:
                        images.append(os.path.join(dirpath, file))
                else:
                    images.append(os.path.join(dirpath, file))
    images.sort()
    return images


def special_print(item, all_num, now, s):
    if now==0:
        print('')
    now = now+1 # for range's speciall consideration
    e = time.time()
    percentage = int(now*100/all_num)
    finish_time = int((e-s)*(all_num-now))
    print("\r  {} |{:<50}| {}% |cost:{:<.3f}s|finished after:{:<4d}s".format(item, '-'*(percentage//2)+'>', percentage, e-s,finish_time), end='\n',
          flush=True)

    if now==all_num:
        print('\n')


def bianry(image):
    image = (np.sign(np.sign(image-0.5)-0.5)+1)/2
    return image


def get_weighted_indices(start, end, num, weights):
    ''' random choice but with weights for different number
        archieved by np.random.choice
    '''
    return np.random.choice(np.arange(start=start, stop=end), size=num,  p=weights)


def special_plot(train_generate):
    for i in train_generate:
        c = []
        for k in range(0,9):
            figure(1)
            subplot(330 + 1 + k)
            image = i[0][k,:,:,:]

            mask = i[1][k,:,:,0]
            mask[where(mask==0)]=None
            c.append(imshow(image/255))
            # title(i[1][k,0])
            # figure(2)
            # subplot(330 + 1 + k)
            c.append(imshow(mask, cmap='Reds_r'))
        pause(0.5)
        # show()
        for k in c:
            k.remove()


def dataset_plot(get_dataset, size,  special_num=0, positive_num=0, random_num=0, noise_num=0):

    while 1:
        c = []
        sample = get_dataset(special_num=special_num, positive_num=positive_num,
                        random_num=random_num, noise_num=noise_num,size=size)
        for k in range(0, 9):
            figure(1)
            subplot(330 + 1 + k)
            image = sample[k, :, :, 0:3]
            mask = sample[k, :, :, 3].astype('float32')
            mask[where(mask==0)]=None
            mask[0,0:2]=[0,1]
            c.append(imshow(image))
            c.append(imshow(mask, cmap='seismic'))

        pause(0.5)
        for k in c:
            k.remove()


def plot_samples(images, mask, pred,  index, image_name):
    w,h = mask.shape[0:2]
    c = np.zeros_like(mask)
    # c[where(mask>0.5)]=1
    # c[where(mask<0.5)]=None
    # c[0, 0:2] = [0, 1]
    c = 2*mask-pred
    c[where(c==0)]=None
    c[0:2,0] = [0,2.5]

    f  = figure(figsize=[h//100,w//100], dpi=100)
    ax = subplot(111)
    imshow(images)
    # imshow(c, cmap='seismic')
    # color={1:'r', 2:'b', 3:'y',0:'k'}
    imshow(c, cmap='seismic')
    color = {1: 'y', 2: 'b', 3: 'b', 0: 'k'}
    for k in index:
        plot(k[1], k[0], color[k[2]], linewidth=2)
        scatter((k[1][0]+k[1][2])/2, (k[0][0]+k[0][2])/2, s=100, marker='*', c=color[k[2]])
    imshow(c, cmap='seismic')
    ylim([0,w])
    xlim([0,h])
    ax.invert_yaxis()
    plt.savefig(image_name, bbox_inches='tight')
    plt.close(f)


def skeleton_test():
    label = Image.open('./drones/top/mix/DJI_0314_label.jpg')
    pred = Image.open('./drones/top/mix/DJI_0314_pred.jpg')
    bear_range = 40
    label = get_label(label)
    # label = np.sign(gaussian(label, 1))
    label = np.sign(mean_blur2d([5,5], label))
    pred = get_label(pred)
    # pred = np.sign(gaussian(pred, 1))
    pred = np.sign(mean_blur2d([5,5], pred))
    skeleton1 = np.sign(skeletonize(label, method='lee'))
    skeleton2 = np.sign(skeletonize(pred, method='lee'))

    print('T length', skeleton1.sum()/100)
    print('P length', skeleton2.sum()/100)

    _, FP, _,_ = confusion_matrix(np.sign(Gaussian_blur2d_faster([bear_range,bear_range], skeleton1)), skeleton2)
    TP,_ ,FN ,_ = confusion_matrix(skeleton1, np.sign(Gaussian_blur2d_faster([bear_range,bear_range], skeleton2)))

    print(TP/100, FP/100)
    print(FN/100)

    figure()
    skeleton2 = np.sign(Gaussian_blur2d_faster([bear_range,bear_range], skeleton2))
    c = np.ones_like(skeleton2) * 0.8
    c[where(skeleton2 == 0)] = None
    c[0, 0:2] = [0,1]
    imshow(c, cmap='seismic')

    skeleton1 = np.sign(Gaussian_blur2d_faster([10, 10], skeleton1))
    c = np.ones_like(skeleton1) * 1.0
    c[where(skeleton1 == 0)] = None
    c[0, 0:2] = [0, 1]
    imshow(c, cmap='gray_r')

    print(confusion_matrix(skeleton1, skeleton2))
    show()


def confusion_matrix(label, pred):
    '''
    compute confusion_matrix
        label: label, binary 0 or 1
        pred: prediction, binary, 0 or 1
        size(label) should be equal to pred

    return 
        [TP, FP, FN, TN]
    '''
    TP = len(where(pred[where(label == 1)] ==1)[0])
    FN = len(where(pred[where(label == 1)] ==0)[0])
    FP = len(where(pred[where(label == 0)] ==1)[0])
    TN = len(where(pred[where(label == 0)] ==0)[0])

    return TP, FP, FN, TN


def get_image_skeleton(image):
    '''
    get skeleton image, from skimage.morphology.skeletonize

    image: 2d binary array
    '''
    return np.sign(skeletonize(image, method='lee'))


def skeleton_confusion_matrix(label, pred, bear_range=40, is_show=False, bg=None, **k):
    '''
    build confusion_matrix after skeleton label and pred\
        bear_range: maxium range when consider pixels from label and pred matched
    '''

    # remove noise
    label = np.sign(Gaussian_blur2d_faster([3,3], label))
    pred = np.sign(Gaussian_blur2d_faster([3,3], pred))
    # print(label.sum())
    # print(pred.sum())
    # get 1 pixel width skeleton
    skeleton1 = np.sign(skeletonize(label, method='lee'))
    skeleton2 = np.sign(skeletonize(pred, method='lee'))

    _, FP, _,_ = confusion_matrix(np.sign(Gaussian_blur2d_faster([bear_range,bear_range], skeleton1)), skeleton2)
    TP,_ ,FN ,_ = confusion_matrix(skeleton1, np.sign(Gaussian_blur2d_faster([bear_range,bear_range], skeleton2)))

    if is_show:
        w,h = label.shape
        figure(figsize=[h//100, w//100], dpi=100)
        if bg is not None:
            imshow(bg)

        # show background
        # skeleton2_thick = np.sign(Gaussian_blur2d_faster([bear_range, bear_range], skeleton2))
        # c = np.ones_like(skeleton2_thick) * 0.8
        # c[where(skeleton2_thick == 0)] = None
        # c[0, 0:2] = [0, 1.5]
        # imshow(c, cmap='bwr_r')

        skeleton2_thin = np.sign(Gaussian_blur2d_faster([15, 15], skeleton2))
        c = np.ones_like(skeleton2_thin) * 1.0
        c[where(skeleton2_thin == 0)] = None
        c[0, 0:2] = [0, 1]
        imshow(c, cmap='bwr_r')

        skeleton1 = np.sign(Gaussian_blur2d_faster([6, 6], skeleton1))
        c = np.ones_like(skeleton1) * 1.0
        c[where(skeleton1 == 0)] = None
        c[0, 0:2] = [0, 1]
        imshow(c, cmap='bwr')
        print('pixel based',confusion_matrix(label, pred))
        print('length based',TP, FP, FN, 0 )
        title(k.get('title'))

        savefig(k.get('title').replace('.JPG', '_all.png'), bbox_inches='tight')
        plt.close('all')

    return TP/100, FP/100, FN/100, 0


def create_copy_folder():
    old = './DataSet/Al_Ula'
    new = './DataSet/Al_Ula_test'

    sub1 = []
    sub2 = []
    sub3 = []

    for k in os.walk(old):
        a,b,c = k
        if a not in sub1:
            sub1.append(a)

    for k in sub1:
        os.mkdir(k.replace(old, new))

def half_size(data):
    '''
    half size data to its half
        data: 3D array or 4D array, n,h,w,c = data.shape, n==1
    return:
        a 4D array, shape= [4, h//2, w//2, c]
    '''
    if len(data.shape)==3:
        h,w,c = data.shape
    else:
        n, h,w,c = data.shape
    new = np.reshape(data, [2,h//2, 2, w//2, c])
    new = new.transpose([0,2,1,3,4])
    new = new.reshape([4, h//2, w//2, c])
    return new


def half_size_reverse(data):
    '''
    recoved halved data to its original size
        data: 4D array, n,h,w,c = data.shape, n==4

    return:
        a 4D array, shape= [1, h*2, w*2, c]
    '''
    n, h, w, c = data.shape
    new = np.reshape(data, [2, 2 , h, w,c])
    new = new.transpose([0, 2, 1, 3, 4])
    new = np.reshape(new, [1, h*2, w*2, c])
    return new


def confusion_matrix_print(c_matrix, item=None):
    TP, FP, FN, TN = c_matrix
    s = '''
                \r###########################################
                \r#{}
                \r#                    input
                \r#Pred          crack          Nc
                \r#crack         {}         {}
                \r#   NC         {}         {}
                \r###########################################
                \r'''.format(item, TP, FP, FN, TN)
    print(s)


def oriention_statistic(name = 'DJI_0872_labell', ONLY_LOAD=True): 
    '''
    calculate orientation by rose diagrams
    name: file name in ./drones, file_path ='./drones/{}.JPG'.format(name)
    '''

    # read data
    path = './drones/gigapans/{}.png'.format(name)
    a = Image.open(path)
    img = get_label(a)
    print(img.shape)
    f_w, f_h = 40,40
    f_width = 5

    # get sekeltonized label
    img_ske = get_image_skeleton(img)

    slips = []
    results = []
    filters = []

    # calculate filter along different directions
    figure(1, figsize=[15,15])
    for k, angle in enumerate(np.linspace(-90, 90, 25)):
        filter = np.zeros([f_w,f_h])

        if abs(angle)==90:
            filter[f_w//2-f_width//2:f_w//2+f_width//2,:] = 1
        else:
            line = lambda x: np.tan(angle/180*np.pi)*(x-f_w//2)+f_h//2
            width = min(100, f_width/(np.cos(angle/180*np.pi)+0.01)) 
            for i in range(f_w):
                middle = line(i)
                s = max(0, int(middle-width//2))
                e = min(f_h, int(middle+width//2))
                if e>s:
                    filter[i,s:e]=1
        subplot(5,5,k+1)
        title('{:.1f}deg'.format(angle+90))
        filter[0,0:2]=[0,1]
        imshow(filter, cmap='gray')
        print(angle,'deg filter generating ')

        filters.append(filter)
        slips.append(angle)
    plt.savefig('./test/filters.jpg')

    # calculate correlation between filters and label

    if not ONLY_LOAD: # muti processes calculation
        s = time.time()
        tasks = []
        with ProcessPoolExecutor(25) as executor:
            for i in range(25):
                filter  = filters[i]
                task = executor.submit(signal.correlate2d, **{'in1':img_ske,'in2':filter, 'boundary':'fill', 'mode':'same'})
                tasks.append(task)
        e = time.time()
        print('\n convolution process finished after {}s'.format(e-s))

        # print statistics
        figure(2, figsize=[15,15])
        for k, task in enumerate(tasks):
            results.append(task.result())
            coor = results[k].copy()
            subplot(5,5,k+1)
            title(slips[k])
            imshow(coor)
            coor[where(coor<10)]=0
            print(slips[k], coor.sum())

        plt.savefig('./test/results.jpg')
        np.save('test/results_{}.npy'.format(name), np.array(results))
    else: # only read computed result
        results = np.load('test/results_{}.npy'.format(name))
        figure(2, figsize=[40,40])
        data = []
        for k in range(results.shape[0]):
            coor = results[k].copy()
            coor[where(coor<20)]=0
            coor[where(coor>0)]=1
            figure(2)
            subplot(5,5,k+1)
            title('{:f}deg'.format(slips[k]+90))
            imshow(coor, cmap='Reds')
            xticks([])
            yticks([])
            print('{:>04},{:>08}'.format(slips[k], int(coor.sum())))
            data.append([slips[k],int(coor.sum())])
        print(coor.shape)
        plt.savefig('./test/results.jpg')
        
        plt.figure(figsize=(5,5), dpi=200)
        ax = plt.subplot(111, projection='polar')
        for angle, coor in data:
            radii = coor/40 # 每一行数据
            ax.bar((angle+90)/180*np.pi, radii, width=np.pi/len(data), color='k')
            ax.bar(np.pi+(angle+90)/180*np.pi, radii, width=np.pi/len(data), color='k')
            yticks([])
            # xticks([0,np.pi/4,np.pi/2,np.pi/4*3,np.pi])
        plt.savefig('test/rose_{}.png'.format(name), bbox_inches='tight', transparent='True')

    return None


if __name__ == "__main__":
    if len(sys.argv) >= 2:
        name = sys.argv[1]
    else:
        name = 'DJI_0872_labell'
        
    if len(sys.argv) == 3:
        ONLY_LOAD=sys.argv[2]
    else:
        ONLY_LOAD=0
    oriention_statistic(name, ONLY_LOAD)