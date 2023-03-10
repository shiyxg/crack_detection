# crack_detection
To run it online: 
https://colab.research.google.com/drive/1n7K1esyM_FmWb-6ZwT8eGWxylOk4xoOp

## environments

* Tensorflow 2
* Kears 2
* h5py=2.10.0

## scripts introduction

### 1 prepare data sets to training a model

* `get_datasets.py` form dataset from images and labels
* `datasets.py` dataset classes definition, to offer package of image reading and picking

### 2 define network's structure,  train a model based on a data set and save it

* `graph.py`: build graphs and models by keras
* `callbacks.py`: self-own callbacks for the trainning
* `trainCNN.py`: train a CNNs network
* `task.sbatch`: sbatch file for ibex to run trainCNN.py

### 3 examine the performance of a trained network on valid set by statistics, eg:offer confusion matrix

* `statistics.py`: get the network's confusion matrix and so on in testset
* `tools.py`: some useful tools: confusion matrix methods and so on


### 4 run good network on all 20k images

* `do_test.py`: use stored parameter to build model and detect cracks in
  an image

### some other aguments tools

* `reduce_small.py`: post-precessing, to reduce the areas without enough
  positive labels
  