import os
for dirpath, dirnames, filenames in os.walk('./'):
    for file in filenames:
        if 'JPG' in file:
            label = file.replace('.JPG','_label.png')
            flag=0
            if label in filenames:
                flag=1
            
            if not flag:
                # print(os.path.join(dirpath, file))
                # name = os.path.join(dirpath, file)
                os.remove(os.path.join(dirpath, file))
             
            #os.rename(name, name.replace('samples', 'Dan_01_samples'))
