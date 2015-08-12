import pickle
import numpy as np
import xml.etree.ElementTree as ET
import os
from random import randint
from skimage.transform import resize
from skimage.io import imread
from skimage.io import imsave
import argparse


def get_class(obj):
    if obj == 'person':
        return 1
    elif obj == 'bird':
        return 2
    elif obj == 'cat':
        return 3
    elif obj == 'cow':
        return 4
    elif obj == 'dog':
        return 5
    elif obj == 'horse':
        return 6
    elif obj == 'sheep':
        return 7
    elif obj == 'aeroplane':
        return 8
    elif obj == 'bicycle':
        return 9
    elif obj == 'boat':
        return 10
    elif obj == 'bus':
        return 11
    elif obj == 'car':
        return 12
    elif obj == 'motorbike':
        return 13
    elif obj == 'train':
        return 14
    elif obj == 'bottle':
        return 15
    elif obj == 'chair':
        return 16
    elif obj == 'diningtable':
        return 17
    elif obj == 'pottedplant':
        return 18
    elif obj == 'sofa':
        return 19
    elif obj == 'tvmonitor':
        return 20


def start(file_num, cropSize):

    path = 'VOCdevkit/VOC2012/'

    set_image = []
    set_bndbox = []
    set_class = []
    size = 1

    f = open('textVOC/'+file_num+'.txt', 'r')
    string = f.read().split()
 
    for num in range(len(string)):
        
        filename = string[num]
        
        image = imread(path+'JPEGImages/'+filename)
        tree = ET.parse(path+'Annotations/'+filename.split('.')[0] + '.xml')
        root = tree.getroot()
        label_bndbox = []
        label_class = []
        
        for obj in root.findall('object'):  # read object in xml files

            name = obj.find('name').text
            bndbox = obj.find('bndbox')

            xmax = bndbox.find('xmax').text
            ymax = bndbox.find('ymax').text
            xmin = bndbox.find('xmin').text
            ymin = bndbox.find('ymin').text
        
            cls = get_class(name)
            
            coord = [int(xmax), int(ymax), int(xmin), int(ymin)]
            label_bndbox.append(coord)
            label_class.append(cls)
        
    
        image, label_bndbox = resizeToFit(image, label_bndbox, cropSize, 1.1) # resize image and bounding box
        
        if (image.shape[0]==cropSize and image.shape[1]==cropSize): # check condition if image is already 200x200

            matrix_image = image.transpose((2,0,1))
            matrix_class = np.zeros((50))
            matrix_bndbox = np.zeros((50,4))

            for z in range(len(label_class)):
                matrix_class[z] = label_class[z]
                matrix_bndbox[z] = label_bndbox[z]
            
            set_image.append(matrix_image)                  
            set_class.append(matrix_class)
            set_bndbox.append(matrix_bndbox)
            
        else: # if image size exceed 200x200
       
            for resized in pyramid(image, scale=1.2, minSize=(cropSize, cropSize)): # for each resized image

                if (resized.shape != image.shape):

                    for k in range(0,len(label_bndbox)): # resize bounding box

                        label_bndbox[k] = [int(min(label_bndbox[k][0]/1.2, resized.shape[1])), int(min(label_bndbox[k][1]/1.2, resized.shape[0])), int(label_bndbox[k][2]/1.2), int(label_bndbox[k][3]/1.2)]

                for i in range(30): # generate 30 image per resized image

                    temp_image, temp_bndbox, temp_class = generate(resized, label_bndbox, label_class, cropSize) # generate image

                    matrix_image = temp_image.transpose((2,0,1))   

                    matrix_class = np.zeros((50))
                    matrix_bndbox = np.zeros((50,4))

                    for z in range(len(temp_class)):
                        matrix_class[z] = temp_class[z]
                        matrix_bndbox[z] = temp_bndbox[z]
                    
                    set_image.append(matrix_image)                  
                    set_class.append(matrix_class)
                    set_bndbox.append(matrix_bndbox)

        size = size + 1    
    
    print 'Numpy Converting'
    set_image = np.array(set_image, dtype=np.dtype('uint8'))
    set_class = np.array(set_class, dtype=np.dtype('uint8'))
    set_bndbox = np.array(set_bndbox, dtype=np.dtype('uint8'))
    
    print 'Numpy Saving'
    np.save('numpy/image_numpy_'+file_num.zfill(2), set_image)
    np.save('numpy/class_numpy_'+file_num.zfill(2), set_class)
    np.save('numpy/bndbox_numpy_'+file_num.zfill(2), set_bndbox)
   
    print "Process Complete"                    

        

def checkExceedBndBox(label_bndbox, cropSize):
    count = 0
    exceed = False
    for coord in label_bndbox:    
        if (coord[0] - coord[2] > cropSize or coord[1] - coord[3] > cropSize):
            count = count + 1
    if (count == len(label_bndbox)):
        exceed = True
    return exceed

def resizeToFit(image, label_bndbox, cropSize, scale):
    exceed = checkExceedBndBox(label_bndbox, cropSize)
    newcoord = np.array(label_bndbox)

    if (image.shape[0] < cropSize or image.shape[1] < cropSize):
        X = image.shape[1] - cropSize
        Y = image.shape[0] - cropSize
        image = resize(image, (cropSize, cropSize))
        xycoord = [X,Y,X,Y]
        newcoord[:] = newcoord[:] - xycoord #calculate new coordinate according to x, y changes
        newcoord = np.clip(newcoord, 0, cropSize-1)
        exceed = False

    while (exceed):
        #w = int(image.shape[1] / scale)
        #res = imutils.resize(image, width=w)
        w = int(image.shape[1] / scale)
        h = int(image.shape[0] / scale)
        res = resize(image, (h, w))
        if (res.shape[0] <= cropSize or res.shape[1] <= cropSize): #image size is too small
            if (res.shape[0] > res.shape[1]):
                #res = imutils.resize(image, width=cropSize)
                ratio = float(image.shape[1])/cropSize
                w = cropSize
                h = int(image.shape[0] / ratio)
                res = resize(image, (h, w))
            else:
                #res = imutils.resize(image, height=cropSize)
                ratio = float(image.shape[0])/cropSize
                w = int(image.shape[1] / ratio)
                h = cropSize
                res = resize(image, (h, w))
            newcoord = newcoord[:,:]/ratio #calculate new coordinate according to resized ratio
            newcoord = newcoord.astype(int)
            exceed = checkExceedBndBox(newcoord, cropSize)
            if (exceed or res.shape[0] < cropSize or res.shape[1] < cropSize):
                X = res.shape[1] - cropSize
                Y = res.shape[0] - cropSize
                res = resize(res, (cropSize, cropSize))
                xycoord = [X,Y,X,Y]
                newcoord[:] = newcoord[:] - xycoord #calculate new coordinate according to x, y changes
                newcoord = np.clip(newcoord, 0, cropSize-1)
                exceed = False
        else:
            
            newcoord = newcoord[:,:]/scale #calculate new coordinate according to resized ratio
            newcoord = newcoord.astype(int) 
            exceed = checkExceedBndBox(newcoord, cropSize)

        #make sure that the coordinates do not exceed the image size
        image = res 
        newcoord[:,0] = np.minimum(newcoord[:,0], image.shape[1]-1)
        newcoord[:,1] = np.minimum(newcoord[:,1], image.shape[0]-1)

    return (image, newcoord)

def generate(image, coordinate, cls, cropSize):
    
    rect = image.copy()
    newcls = []
    newcoord = []
    notDone = True
    coordOk = []
    checkInRange = [False,False,False,False]

    while (notDone):

        randY = randint(0,image.shape[0]-cropSize)
  
        randX = randint(0,image.shape[1]-cropSize)
        newcoord = []
        newcls = []
        
        tempCoord = [0,0,0,0]
        checkInRange[0] = np.less(coordinate[:,0], randX)
        checkInRange[1] = np.less(coordinate[:,1], randY)
        checkInRange[2] = np.greater(coordinate[:,2], randX+cropSize)
        checkInRange[3] = np.greater(coordinate[:,3], randX+cropSize)
        
        for i in range(0,len(coordinate)):
            if not (np.all([checkInRange[0][i],checkInRange[1][i],checkInRange[2][i],checkInRange[3][i]])):
                tempCoord = [coordinate[i,0]-randX, coordinate[i,1]-randY, coordinate[i,2]-randX, coordinate[i,3]-randY]
                tempCoord = np.clip(tempCoord, 0, cropSize-1)
                cropObjArea = (tempCoord[0] - tempCoord[2])*(tempCoord[1] - tempCoord[3])
                bndBoxArea = (coordinate[i,0] - coordinate[i,2])*(coordinate[i,1] - coordinate[i,3])*0.7
                if (cropObjArea >= bndBoxArea):
                    newcls.append(cls[i])
                    newcoord.append([tempCoord[0],tempCoord[1],tempCoord[2],tempCoord[3]])
                          
        if (len(newcoord) > 0):
            subimg = image[randY:randY+cropSize,randX:randX+cropSize,:]
            notDone = False
    
    return (subimg, newcoord, newcls)

def pyramid(image, scale=1.2, minSize=(200, 200)):
    # yield the original image
    yield image
 
    # keep looping over the pyramid
    while True:
        # compute the new dimensions of the image and resize it
        w = int(image.shape[1] / scale)
        h = int(image.shape[0] / scale)
        image = resize(image, (h, w))
        #image = imutils.resize(image, width=w)
 
        # if the resized image does not meet the supplied minimum
        # size, then stop constructing the pyramid
        if image.shape[0] < minSize[1] or image.shape[1] < minSize[0]:
            break
 
        # yield the next image in the pyramid
        yield image




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--text', type=str, required=True)
    args = parser.parse_args()
    start(args.text, 200)
