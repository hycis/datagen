import os
import sys

if __name__ == "__main__":
    file_size = sys.argv[1]


    path = 'VOCdevkit/VOC2012/'

    num_file = 1
    size = 1

    for filename in os.listdir(path+'JPEGImages/'):
        

        f = open('text/'+str(num_file)+'.txt', 'a')
        f.write(filename+'\n')
        
        f.close()

        if size % int(file_size) == 0:
            num_file = num_file + 1
        size = size + 1
        

