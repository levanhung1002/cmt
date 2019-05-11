import cv2
import numpy as np
import os
import sys
import opencvYolo
from pydarknet import Detector, Image
from tkinter import filedialog
from tkinter import *
import json
def sortLine(e):
    return e[0][0]

def rotate_bound(image, angle):
    # grab the dimensions of the image and then determine the
    # center
    (h, w) = image.shape[:2]
    (cX, cY) = (w // 2, h // 2)
 
    M = cv2.getRotationMatrix2D((cX, cY), -angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])
 
    # compute the new bounding dimensions of the image
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))
 
    # adjust the rotation matrix to take into account translation
    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY
 
    # perform the actual rotation and return the image
    return (cv2.warpAffine(image, M, (nW, nH)), M)

def getPoint(src, M):
    x, y = src
    return(M[0][0]*x + M[0][1]*y + M[0][2], M[1][0]*x + M[1][1]*y + M[1][2])

def getArrPoint(src, M):
    dst = list()
    for point in src:
        x, y = point
        dst.append((M[0][0]*x + M[0][1]*y + M[0][2], M[1][0]*x + M[1][1]*y + M[1][2]))
    return dst

def getData(direction):
    data = dict()
    for box in direction:
        if box[0] in data:
            if data[box[0]][0] < box[1]:
                data[box[0]] = [box[1], box[2][0], box[2][1], box[2][2], box[2][3]]
        else:
            data[box[0]] = [box[1], box[2][0], box[2][1], box[2][2], box[2][3]]
    return data

    


def lableName(image, configPath = "./config/yolo-obj-word.cfg", \
    weightPath = "./weight/yolo-obj_word.weights" , \
    metaPath = "./meta/word.names"):
    direction = opencvYolo.performDetect(image, 0.25, configPath, weightPath, metaPath)
    line = []
    line1 = []

    for box in direction:
        if(box[0] == 'chu' and box[1] >= 0.5):
            x1 = box[2][0] - box[2][2]/2
            y1 = box[2][1] - box[2][3]/2
            x2 = box[2][0] + box[2][2]/2
            y2 = box[2][1] + box[2][3]/2
            line.append([[x1, y1], [x2, y2]])
    line.sort(key=sortLine)
    Y = 0
    for i in range(0,len(line)-1):
        if(i == 0):
            Y = (line[i][0][1] + line[i][1][1])
        if(abs(Y - line[i + 1][0][1] - line[i + 1][1][1]) <= 40):
            Y = line[i + 1][0][1] + line[i + 1][1][1]
        else:
            line1.append(line[i + 1])

    if(len(line1) != 0):
        for i in line1:
            if i in line:
                line.remove(i)

    for i in range(0, len(line) - 1):
        tb = (line[i][1][0] + line[i + 1][0][0])/2
        line[i][1][0] = tb
        line[i + 1][0][0] = tb

    for i in range(0, len(line1) - 1):
        tb = (line1[i][1][0] + line1[i + 1][0][0])/2
        line1[i][1][0] = tb
        line1[i + 1][0][0] = tb
    if(len(line) != 0):
        line[len(line) - 1][1][0] = min(line[len(line) - 1][1][0] + 5 , 560)
    if(len(line1) != 0):
        line1[len(line1) - 1][1][0] = min(line1[len(line1) - 1][1][0] + 5 , 560)
        if(line[0][0][1] + line[0][1][1] - line1[0][0][1] - line1[0][1][1]) < 0:
            line.extend(line1)
        else:
            line1.extend(line)
            line = line1
    line = np.array(line, np.int32)
    images = list()
    
    arrNames = []
    for i in line:
        img = image[i[0][1]:i[1][1], i[0][0]:i[1][0]]
        images.append(img)
        arrNames.append(np.array([(i[0][0] + 265, i[0][1] + 180),(i[1][0] + 265, i[0][1] + 180), (i[1][0] + 265,i[1][1] + 180), (i[0][0] + 265, i[1][1] + 180)], np.float32))
    return (images, arrNames)


def idDetect(image, configPath = "./config/id.cfg" , \
    weightPath = "./weight/id.weights", \
    metaPath = "./meta/id.names"):

    direction = opencvYolo.performDetect(image, 0.25, configPath, weightPath, metaPath)
    preId = 0
    imgId = None
    for box in direction:
        if(box[1] >= preId):
            x1 = box[2][0] - box[2][2]/2
            y1 = box[2][1] - box[2][3]/2
            x2 = box[2][0] + box[2][2]/2
            y2 = box[2][1] + box[2][3]/2
            preID = box[1]
            imgId = image[int(y1): int(y2), int(x1):int(x2)]

    return imgId

def getDataYolo(path):
    f = open(path, 'r')
    data = json.load(f)
    return data

def writeYoloAdd(img, pathYolo, data, coor):
    h, w = img.shape[:2]

    # coor = [(w*0.47, 115), (w, 190)]
    x, y = coor
    file = open(pathYolo, "a")
    maxHeight = max([i[1][1] - i[0][1] for i in data])

    for i in data:
        x_center = (i[0][0]*0.5 + i[1][0]*0.5 - x)/w
        y_center = (i[0][1]*0.5 + i[1][1]*0.5 - y)/h
        w_center = (i[1][0] - i[0][0])/w
        h_center = (i[1][1] - i[0][1])/h
        height = i[1][1] - i[0][1]
        if(height < 0.5*maxHeight):
            file.writelines(str(1) + " " + str.format('{0:.6f}',x_center) + " " + str.format('{0:.6f}',y_center) + " " + str.format('{0:.6f}',w_center) + " " + str.format('{0:.6f}',h_center) + '\n')
        else:
            file.writelines(str(0) + " " + str.format('{0:.6f}',x_center) + " " + str.format('{0:.6f}',y_center) + " " + str.format('{0:.6f}',w_center) + " " + str.format('{0:.6f}',h_center) + '\n')
    file.close()

def writeYolo(img, pathYolo, data, coor):
    h, w = img.shape[:2]

    # coor = [(w*0.47, 115), (w, 190)]
    x, y = coor
    file = open(pathYolo, "a")

    for i in data:
        x_center = (i[0][0]*0.5 + i[1][0]*0.5 - x)/w
        y_center = (i[0][1]*0.5 + i[1][1]*0.5 - y)/h
        w_center = (i[1][0] - i[0][0])/w
        h_center = (i[1][1] - i[0][1])/h
        file.writelines(str(0) + " " + str.format('{0:.6f}',x_center) + " " + str.format('{0:.6f}',y_center) + " " + str.format('{0:.6f}',w_center) + " " + str.format('{0:.6f}',h_center) + '\n')
    file.close()

def transformImg(image, file, output, dataYolo, configPath = "./config/cmcc.cfg" , \
    weightPath = "./weight/cmcc.weights", \
    metaPath = "./meta/cmcc.data"):

    (h, w) = image.shape[:2]
    img = image.copy()

    net = Detector(bytes(configPath, encoding="utf-8"), bytes(weightPath, encoding="utf-8"), 0, bytes(metaPath,encoding="utf-8"))
    img_darknet = Image(image)

    direction = net.detect(img_darknet)

    # direction = opencvYolo.performDetect(image, 0.25, configPath, weightPath, metaPath)
    data = getData(direction)

    if('can_2' in data or 'va_2' in data or 'thang_2' in data \
        or 'nam_2' in data or 'gioi_2' in data or 'tinh_2' in data \
        or 'quoc_2' in data or 'tich_2' in data or 'que_2' in data):

        # can cuoc cong dan
        #id
        try:
            id_x1 = ['so_1', 'ten_1']
            x1_id = int(max([data[i][1] + data[i][3]/2 for i in id_x1 if i in data]))
            id_y1 = ['can_2', 'dan_1']
            y1_id = int(min([data[i][2] + data[i][4]*0.4 for i in id_y1 if i in data]))
            id_y2 = ['ho_1', 'ten_1', 'va_2']
            y2_id = int(max([data[i][2] - data[i][4]/2 for i in id_y2 if i in data]))

            idImg = img[y1_id:y2_id, x1_id:w]
            
            cv2.rectangle(image, (x1_id, y1_id), (w, y2_id), (0, 20, 100), 2)

            #name

            name_x1 = ['ho_1', 'ngay_1', 'gioi_2', 'que_2', 'noi_1']
            x1_name = int(min([data[i][1] - data[i][3]/2 for i in name_x1 if i in data]))
            if 'so_1' in data:
                name_y1 = ['so_1']
                y1_name = int(min([data[i][2] + data[i][4]/2 for i in name_y1 if i in data]))
            else:
                y1_name = int((y2_id + y1_id)*0.53)
            name_y2 = ['sinh_1', 'ngay_1', 'thang_2', 'nam_2']
            y2_name = int(max([data[i][2] - data[i][4]/2 for i in name_y2 if i in data]))
            nameImg = img[y1_name:y2_name, x1_name:w]
            
            cv2.rectangle(image, (x1_name, y1_name), (w, y2_name), (0, 200, 200), 2)

            # #ngay sinh

            date_x1 = ['sinh_1', 'ngay_1', 'thang_2', 'nam_2']
            x1_date = int(max([data[i][1] + data[i][3]/2 for i in date_x1 if i in data]))
            y1_date = int((y2_id + y2_name)*0.53)
            date_y2 = ['gioi_2', 'tinh_2', 'quoc_2', 'tich_2']
            y2_date = int(max([data[i][2] - data[i][4]/2 for i in date_y2 if i in data]))
            dateImg = img[y1_date:y2_date, x1_date:w]
            
            cv2.rectangle(image, (x1_date, y1_date), (w, y2_date), (0, 200, 20), 2)
                # nguyen quan

            home_x1 = ['ho_1', 'ngay_1', 'gioi_2', 'que_2', 'noi_1']
            x1_home = int(min([data[i][1] - data[i][3]/2 for i in home_x1 if i in data]))
            home_y1 = ['gioi_2', 'tinh_2', 'quoc_2', 'tich_2']
            y1_home = int(min([data[i][2] + data[i][4]/3 for i in home_y1 if i in data]))
            home_y2 = ['noi_1', 'thuong_1', 'tru_1']
            y2_home = int(max([data[i][2] - data[i][4]/2 for i in home_y2 if i in data]))
            homeImg = img[y1_home:y2_home, x1_home:w]
            
            cv2.rectangle(image, (x1_home, y1_home), (w, y2_home), (100, 20, 200), 2)

            # dia chi
            add_x1 = ['ho_1', 'ngay_1', 'gioi_2', 'que_2', 'noi_1']
            x1_add = int(min([data[i][1] - data[i][3]/2 for i in add_x1 if i in data]))
            y1_add = int((y2_date + y2_home)*0.55)
            addImg = img[y1_add:h, x1_add:w]
            cv2.rectangle(image, (x1_add, y1_add), (w, h), (200, 20, 0), 2)

            cv2.imwrite(output + "/id/" + file, idImg)
            pathId = output + "/id/" + file[:len(file) -4] + ".txt"
            writeYolo(idImg, pathId, [dataYolo['id_rect']], (x1_id, y1_id))

            cv2.imwrite(output + "/name/" + file, nameImg)
            pathName = output + "/name/" + file[:len(file) -4] + ".txt"
            writeYolo(nameImg, pathName, dataYolo['NameWordCoors'], (x1_name, y1_name))

            cv2.imwrite(output + "/date/" + file, dateImg)
            pathDate = output + "/date/" + file[:len(file) -4] + ".txt"
            writeYolo(dateImg, pathDate, [dataYolo['BirthdateCoor']], (x1_date, y1_date))

            cv2.imwrite(output + "/home/" + file, homeImg)
            pathHome = output + "/home/" + file[:len(file) -4] + ".txt"
            writeYoloAdd(homeImg, pathHome, dataYolo['HometownWordCoors'], (x1_home, y1_home))

            cv2.imwrite(output + "/add/" + file, addImg)
            pathAdd = output + "/add/" + file[:len(file) -4] + ".txt"
            writeYoloAdd(addImg, pathAdd, dataYolo['AddressWordCoors'], (x1_add, y1_add))

            return image, True
            
        except ValueError:
            print('Invalid value!')
            return image, False
        except KeyError:
            print('Invalid value!')
            return image, False

    else:
        # chung minh nhan dan

        # th1: cat theo template

        pts = np.empty((0,2))
        dst = np.empty((0,2))

        template = {'cong_1' : [340,42], \
            'giay_1' : [360,110], \
            'doc_1' : [418,70], \
            'nam_1' : [782,42], \
            'phuc_1' : [715,70], \
            'dan_1' : [772,110], \
            'noi_1' : [294,462], \
            'tru_1' : [511,462], \
            'nguyen_1' : [322,368], \
            'thuong_1' : [445,462]}

        try:
            id_x1 = ['so_1', 'giay_1', 'ngay_1', 'dkhk_1']
            x1_id = int(max([data[i][1] + data[i][3]/2 for i in id_x1 if i in data]))
            id_y1 = ['giay_1', 'dan_1']
            y1_id = int(min([data[i][2] + data[i][4]*0.4 for i in id_y1 if i in data]))
            id_y2 = ['ho_1', 'ten_1']
            y2_id = int(max([data[i][2] - data[i][4]/2 for i in id_y2 if i in data]))

            idImg = img[y1_id:y2_id, x1_id:w]
            cv2.rectangle(image, (x1_id, y1_id), (w, y2_id), (0, 20, 100), 2)

                #name

            name_x1 = ['ho_1', 'sinh_1', 'nguyen_1', 'noi_1']
            x1_name = int(min([data[i][1] - data[i][3]/2 for i in name_x1 if i in data]))
            if 'so_1' in data:
                name_y1 = ['so_1']
                y1_name = int(min([data[i][2] + data[i][4]/2 for i in name_y1 if i in data]))
            else:
                y1_name = int((y2_id + y1_id)*0.53)
            name_y2 = ['sinh_1', 'ngay_1']
            y2_name = int(max([data[i][2] - data[i][4]/2 for i in name_y2 if i in data]))
            nameImg = img[y1_name:y2_name, x1_name:w]
            cv2.rectangle(image, (x1_name, y1_name), (w, y2_name), (0, 200, 200), 2)

                # #ngay sinh

            date_x1 = ['ngay_1', 'dkhk_1']
            x1_date = int(max([data[i][1] + data[i][3]/2 for i in date_x1 if i in data]))
            y1_date = int((y2_id + y2_name)*0.53)
            date_y2 = ['nguyen_1', 'quan_1']
            y2_date = int(max([data[i][2] - data[i][4]/2 for i in date_y2 if i in data]))
            dateImg = img[y1_date:y2_date, x1_date:w]
            cv2.rectangle(image, (x1_date, y1_date), (w, y2_date), (0, 200, 20), 2)
                # nguyen quan

            home_x1 = ['ho_1', 'sinh_1', 'nguyen_1', 'noi_1']
            x1_home = int(min([data[i][1] - data[i][3]/2 for i in home_x1 if i in data]))
            home_y1 = ['sinh_1', 'ngay_1']
            y1_home = int(min([data[i][2] + data[i][4]/3 for i in home_y1 if i in data]))
            home_y2 = ['noi_1', 'dkhk_1', 'thuong_1', 'tru_1']
            y2_home = int(max([data[i][2] - data[i][4]/2 for i in home_y2 if i in data]))
            homeImg = img[y1_home:y2_home, x1_home:w]
            cv2.rectangle(image, (x1_home, y1_home), (w, y2_home), (100, 20, 200), 2)

                # dia chi
            # add_x1 = ['ho_1', 'sinh_1', 'nguyen_1', 'noi_1']
            # x1_add = int(min([data[i][1] - data[i][3]/2 for i in add_x1 if i in data]))
            y1_add = int((y2_date + y2_home)*0.53)
            addImg = img[y1_add:h, 0:w]
            cv2.rectangle(image, (0, y1_add), (w, h), (200, 20, 0), 2)

            cv2.imwrite(output + "/id/" + file, idImg)
            pathId = output + "/id/" + file[:len(file) -4] + ".txt"
            writeYolo(idImg, pathId, [dataYolo['id_rect']], (x1_id, y1_id))

            cv2.imwrite(output + "/name/" + file, nameImg)
            pathName = output + "/name/" + file[:len(file) -4] + ".txt"
            writeYolo(nameImg, pathName, dataYolo['NameWordCoors'], (x1_name, y1_name))

            cv2.imwrite(output + "/date/" + file, dateImg)
            pathDate = output + "/date/" + file[:len(file) -4] + ".txt"
            writeYolo(dateImg, pathDate, [dataYolo['BirthdateCoor']], (x1_date, y1_date))

            cv2.imwrite(output + "/home/" + file, homeImg)
            pathHome = output + "/home/" + file[:len(file) -4] + ".txt"
            writeYoloAdd(homeImg, pathHome, dataYolo['HometownWordCoors'], (x1_home, y1_home))

            cv2.imwrite(output + "/add/" + file, addImg)
            pathAdd = output + "/add/" + file[:len(file) -4] + ".txt"
            writeYoloAdd(addImg, pathAdd, dataYolo['AddressWordCoors'], (0, y1_add))

            return image, True
        except ValueError:
            print('Invalid value!')
            return image, False
        except KeyError:
            print('Invalid value!')
            return image, False
        
    return None, False

def tranformFolder(input, output, data):
    if(os.path.isdir(input)):
        if not os.path.exists(output + "/id"):
            os.makedirs(output + "/id")
        if not os.path.exists(output + "/name"):
            os.makedirs(output + "/name")
        if not os.path.exists(output + "/date"):
            os.makedirs(output + "/date")
        if not os.path.exists(output + "/home"):
            os.makedirs(output + "/home")
        if not os.path.exists(output + "/add"):
            os.makedirs(output + "/add")
        if not os.path.exists(output + "/draw"):
            os.makedirs(output + "/draw")
        for file in os.listdir(data):
            if '.CMT.' in file:
                # print(file)
                image = cv2.imread(input + "/" + file[:len(file) -4] + ".jpg")
                dataYolo = getDataYolo(data + "/" + file)
                # warped, imgId, imgName = transformImage(image)
                wrap, flag = transformImg(image, file[:len(file) -4] + ".jpg", output, dataYolo)
                if wrap is not None:
                    cv2.imwrite(output + "/draw/" + file[:len(file) -4] + ".jpg", wrap)
                if flag == False:
                    print(file)
                # if warped is None or imgId is None or imgName is None:
                #     continue
                
                # cv2.imwrite(output + "/names/" + file[0:len(file) - 4] + ".NAME.jpg", imgName)
    elif(os.path.isfile(input)):
        return

    # opencvYolo.net = None
    # if(os.path.isdir(output + "/names")):
    #     for file in os.listdir(output + "/names"):
    #         if '.CMT.NAME.' in file:
    #             image = cv2.imread(output + "/names/" + file)
    #             imgs, arrName = lableName(image)
    #             if imgs is None:
    #                 continue
    #             if not os.path.exists(output + "/name/" + file[0:len(file) - 13]):
    #                 os.makedirs(output + "/name/" + file[0:len(file) - 13])
    #             for i in range(0,len(imgs)):
    #                 cv2.imwrite(output + "/name/" + file[0:len(file) - 13] + "/" + str(i) + ".jpg", imgs[i])
    # elif(os.path.isfile(output + "/names")):
    #     return

    # opencvYolo.net = None
    # if(os.path.isdir(output + "/id")):
    #     for file in os.listdir(output + "/id"):
    #         if '.CMT.ID.' in file:
    #             image = cv2.imread(output + "/id/" + file)
    #             imgs = idDetect(image)
    #             if imgs is None:
    #                 continue
    #             if not os.path.exists(output + "/idcut"):
    #                 os.makedirs(output + "/idcut")
    #             cv2.imwrite(output + "/idcut/" + file, imgs)
    # elif(os.path.isfile(output + "/idcut")):
        # return


def main():
        root = Tk()
        root.directory = filedialog.askdirectory(title='Select INPUT folder')
        input_folder = str(root.directory)
        root.directory = filedialog.askdirectory(title='Select SAVE folder')
        output_folder = str(root.directory)
        root.directory = filedialog.askdirectory(title='Select Data folder')
        data_folder = str(root.directory)
        root.destroy()
        if input_folder == '()':
            print('You must select a folder')
            return
        tranformFolder(input_folder, output_folder, data_folder)  

main()
