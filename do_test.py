from graph import UNET, UNET_VGG16, UNET_RESNET
from pylab import *
from PIL import Image
import os
from tools import get_all_images, SaveLabel, special_print,save_label_image
from tools import half_size_reverse, half_size

os.environ['KMP_WARNINGS'] = '0'

#image_dir = 'DataSet/Al_Ula/Resort/High_Altitude'
#label_dir = 'DataSet/Al_Ula_labelled/Resort/High_Altitude'
#npy_dir = 'DataSet/Al_Ula_npy/Resort/High_Altitude'
#appendix = '.JPG'
from_npy = False
#image_dir = 'drones/top_new_label'
#label_dir = 'drones/top_new_label'
#npy_dir = 'drones/top_new_label'
#appendix = '_05_pred.jpg'

image_dir = 'drones/top_new_label'
label_dir = 'drones/top_new_label'
npy_dir = 'drones/top_new_label'
appendix = '_dan_02_pred.jpg'
appendixNPY = '_dan_02_pred.npy'                                                                                                  

save_label = SaveLabel(save_raw=True, save_image=True,
                       thre = 0.5, appendixJPG = appendix, appendixNPY =appendixNPY,
                       works_num = 1,
                       dirs = [image_dir, label_dir, npy_dir])
image_friend = None
thre = []
weight = 'logs/top_dan_01/model_check_point_latest'
network = UNET_RESNET(INPUT_SHAPE=[1504, 2000, 3], class_num=2)
pre_process = lambda x : x.astype('float32')
# pre_process = lambda x: (x.astype('float32')-128)/128

network.do_BN = True
network.do_dropout = True
network.do_transpose = False
model = network.build_network(use_focal_loss=True)
model.load_weights(weight)

drones = get_all_images(image_dir, wanted='.JPG', friend=image_friend)
print('begin running',flush=True)
for i in range(len(drones[:])):
    s = time.time()
    image_name = drones[i]
    image_raw = Image.open(image_name)
    h_r,w_r = image_raw.size
    image = image_raw.resize([4000, 3008])
    image = np.array(image)
    if not from_npy:
        image = half_size(image)
        image_bn = pre_process(image)
        t1 = time.time()
        label = model.predict(image_bn, batch_size=1)
        t2 = time.time()
        label = half_size_reverse(label)[0,:,:,0]
        image = half_size_reverse(image)
    else:
        t1 = time.time()
        npy_name = image_name.replace(image_dir, npy_dir)
        npy_name = npy_name.replace('.JPG', '.npy')
        label = np.load(npy_name).astype('float32')/255
        t2 = time.time()
    # save_label_image(label, image, 'test.jpg', thre=0.5, size=[h_r, w_r])
    save_label.submit_job(label, image, image_name, size=[h_r, w_r])
    t3 = time.time()
    print('read:{:<.3f}s,pred:{:<.3f}s,save:{:<.3f}s'.format(t3-t2, t2-t1, t1-s), flush=True)
    special_print(image_name.replace(image_dir, label_dir), len(drones), i, s)

save_label.wait()
