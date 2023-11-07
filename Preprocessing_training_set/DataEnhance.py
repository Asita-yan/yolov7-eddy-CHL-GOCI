from PIL import Image, ImageEnhance, ImageOps, ImageFile
import numpy as np
import cv2
import random
import threading, os, time
import logging
import xml.etree.ElementTree as ET
import math
import random
import numpy as np
import cv2
import matplotlib.pyplot as plt


Image.MAX_IMAGE_PIXELS = 2300000000

'''
本函数的作用：数据增强
    1. 高斯噪声 gaussian noise 
    2. 椒盐噪声 impulse noise
    3. 图像去噪
    4. 直方图均衡化
    5. 直方图规定化
'''
logger = logging.getLogger(__name__)
ImageFile.LOAD_TRUNCATED_IMAGES = True

class DataAugmentation:

    def __init__(self):
        pass

    @staticmethod
    def openImage(image):
        return Image.open(image, mode="r")

    
    @staticmethod
    def randomGaussian(image, label, mean=0.2, sigma=2):
        """
         对图像添加高斯噪声
        """
        def gaussianNoisy(im, mean=0.2, sigma=2):
            """
            对图像做高斯噪音处理
            :param im: 单通道图像
            :param mean: 偏移量
            :param sigma: 标准差
            :return:
            """
            for _i in range(len(im)):
                im[_i] += random.gauss(mean, sigma)
            return im

        # 将图像转化成数组
        # image= Image.fromarray(image,'RGB')
        img = np.array(image)
        img.flags.writeable = True  # 将数组改为读写模式
        width, height = img.shape[:2]
        img_r = gaussianNoisy(img.flatten(), mean, sigma)
        # img_g = gaussianNoisy(img[:, :, 1].flatten(), mean, sigma)
        # img_b = gaussianNoisy(img[:, :, 2].flatten(), mean, sigma)
        img = img_r.reshape([width, height])
        # img[:, :, 1] = img_g.reshape([width, height])
        # img[:, :, 2] = img_b.reshape([width, height])

        return Image.fromarray(np.uint8(img)), label

    @staticmethod
    def Imagerotate(image, label):
        # 由椭圆的x求y
        def _ellipseX2Y(a, b, x, xc, yc):
            m = math.sqrt(b ** 2 - (x - xc) ** 2 * b ** 2 / a ** 2)
            return yc - m, yc + m

        # 由椭圆的y求x 其实形式上跟上面的函数是一样的 但怕用的时候弄混 还是写了一个函数
        def _ellipseY2X(a, b, y, xc, yc):
            m = math.sqrt(a ** 2 - (y - yc) ** 2 * a ** 2 / b ** 2)
            return xc - m, xc + m

        # 得到矩形框内接椭圆的采样点
        def _getEllipsePt(x1, y1, x2, y2):
            pts = []
            xc = (x1 + x2) / 2
            yc = (y1 + y2) / 2
            a = (x2 - x1) / 2
            b = (y2 - y1) / 2
            # 十字架上的点
            pts.append((x1, yc))
            pts.append((xc, y1))
            pts.append((x2, yc))
            pts.append((xc, y2))
            # 由x求y
            xtmp = x1 + a / 2
            ytmp1, ytmp2 = _ellipseX2Y(a, b, xtmp, xc, yc)
            pts.append((xtmp, ytmp1))
            pts.append((xtmp, ytmp2))
            xtmp = x1 + a / 2 * 3
            ytmp1, ytmp2 = _ellipseX2Y(a, b, xtmp, xc, yc)
            pts.append((xtmp, ytmp1))
            pts.append((xtmp, ytmp2))
            # 由y求x
            ytmp = y1 + b / 2
            xtmp1, xtmp2 = _ellipseY2X(a, b, ytmp, xc, yc)
            pts.append((xtmp1, ytmp))
            pts.append((xtmp2, ytmp))
            ytmp = y1 + b / 2 * 3
            xtmp1, xtmp2 = _ellipseY2X(a, b, ytmp, xc, yc)
            pts.append((xtmp1, ytmp))
            pts.append((xtmp2, ytmp))

            return pts

        def _getRotateXY(x, y, matRotate):
            xnew = matRotate[0][0] * x + matRotate[0][1] * y + matRotate[0][2]
            ynew = matRotate[1][0] * x + matRotate[1][1] * y + matRotate[1][2]
            return xnew, ynew

        # 随机旋转的代码
        # imgFile = r'test.jpg'

        img = np.array(image)

        imgW = img.shape[1]
        imgH = img.shape[0]
        rotateDeg = random.randint(-180, 180)
        rotateScale = 1

        # 旋转矩阵
        matRotate = cv2.getRotationMatrix2D((img.shape[1] * 0.5, img.shape[0] * 0.5), rotateDeg, 1).astype(np.float32)

        # 计算旋转后图像的四个顶点的坐标
        h, w = img.shape[:2]
        corners = np.array([(0, 0), (0, h), (w, 0), (w, h)])
        rotated_corners = cv2.transform(np.array([corners]), matRotate)[0]

        # 计算旋转后图像的最小外接矩形的宽和高
        x, y, w2, h2 = np.array(cv2.boundingRect(rotated_corners))

        # 计算扩充后图像的宽和高
        new_w, new_h = w2, h2
        matRotate2 = cv2.getRotationMatrix2D((new_w * 0.5, new_h * 0.5), rotateDeg, 1).astype(np.float32)
        # 扩充后的图像
        matRotate[0, 2] += (new_w - w) // 2
        matRotate[1, 2] += (new_h - h) // 2
        img = cv2.warpAffine(img, matRotate, (new_w, new_h))



        # # 旋转矩阵
        # matRotate = cv2.getRotationMatrix2D((imgW * 0.5, imgH * 0.5), rotateDeg, rotateScale)
        # # 图像根据旋转矩阵旋转
        # img = cv2.warpAffine(img, matRotate, (imgW, imgH))
        # box取内接椭圆的点
        for obj in label.findall('.//object'):
            box = obj.find('bndbox')
            samplePts = _getEllipsePt(int(box.find('xmin').text), int(box.find('ymin').text), int(box.find('xmax').text), int(box.find('ymax').text))
            # 采样点做旋转
            rotPts = []
            for samplePt in samplePts:
                rotPt = _getRotateXY(samplePt[0], samplePt[1], matRotate)
                rotPts.append(rotPt)
            min_x, min_y = min(rotPts, key=lambda p: p[0])[0], min(rotPts, key=lambda p: p[1])[1]
            max_x, max_y = max(rotPts, key=lambda p: p[0])[0], max(rotPts, key=lambda p: p[1])[1]
            box.find('xmin').text = str(int(min_x))
            box.find('ymin').text = str(int(min_y))
            box.find('xmax').text = str(int(max_x))
            box.find('ymax').text = str(int(max_y))
        # newBox = [x1, y1, x2, y2]
        return Image.fromarray(np.uint8(img)), label
    
    @staticmethod
    def noiseImpulse(image, label, proportion=0.0075):
        """
         对图像添加椒盐噪声
         proportion表示添加椒盐噪声的比例
        """
        # 将图像转化成数组
        # image = Image.fromarray(image, 'RGB')
        img = np.array(image)
        img.flags.writeable = True  # 将数组改为读写模式
        width, height = img.shape[:2]
        num = int(width*height*proportion) # 多少个像素点被添加椒盐噪声
        for i in range(num):
            w = random.randint(0, width - 1)
            h = random.randint(0, height - 1)
            if random.randint(0, 1) == 0:
                img[w, h] = 0
            else:
                img[w, h] = 255
        
        return Image.fromarray(np.uint8(img)), label
    
    @staticmethod
    def imageDenoising(image, label):

        # image = Image.fromarray(image, 'RGB')
        img = np.array(image)
        img.flags.writeable = True  # 将数组改为读写模式
        img = cv2.blur(img, (5,5))

        return Image.fromarray(np.uint8(img)), label
    
    
    @staticmethod
    def histogramEqualization(image, label):

        # image = Image.fromarray(image, 'RGB')
        img = np.array(image)
        img_r = cv2.equalizeHist(img)


        img= img_r

        
        return Image.fromarray(np.uint8(img)), label


    @staticmethod
    def saveImage(image, path):
        image.save(path)

def makeDir(path):
    try:
        if not os.path.exists(path):
            if not os.path.isfile(path):
                # os.mkdir(path)
                os.makedirs(path)
            return 0
        else:
            return 1
    except Exception as e:
        print(str(e))
        return -2


def imageOps(func_name, image, label, img_des_path, label_des_path, img_file_name, label_file_name, times=1):
    funcMap = {"randomGaussian": DataAugmentation.randomGaussian, # 高斯噪声
               "noiseImpulse": DataAugmentation.noiseImpulse, # 椒盐噪声
               "imageDenoising": DataAugmentation.imageDenoising, # 图像去噪
               "histogramEqualization": DataAugmentation.histogramEqualization, # 直方图均衡化
               'Imagerotate':DataAugmentation.Imagerotate
               }

    if funcMap.get(func_name) is None:
        logger.error("%s is not exist", func_name)
        return -1

    for _i in range(0, times, 1):
        new_image, new_label = funcMap[func_name](image, label)
        DataAugmentation.saveImage(new_image, os.path.join(img_des_path, img_file_name[: -4] + "_" + func_name + str(_i) + ".jpg"))
        
        tree = ET.ElementTree(new_label)
        tree.write(os.path.join(label_des_path, label_file_name[: -4] + "_" + func_name + str(_i) + ".xml"))


# 反转、施加噪声等在内的数据扩充操作
opsList = ("randomGaussian", "noiseImpulse", "imageDenoising", "histogramEqualization",'Imagerotate')
# opsList = (['Imagerotate'])

def EnhancedImage(img_path, new_img_path, label_path, new_label_path):

    # img path
    if os.path.isdir(img_path):
        img_names = os.listdir(img_path)
    else:
        img_names = [img_path]

    # label path
    if os.path.isdir(label_path):
        label_names = os.listdir(label_path)
    else:
        label_names = [label_path]
    
    img_num = 0
    label_num = 0

    # statistics img num
    for img_name in img_names:
        tmp_img_name = os.path.join(img_path, img_name)
        if img_name[-4]!='.':
            continue
        else:
            img_num = img_num + 1;
    # statistics label num
    for label_name in label_names:
        tmp_label_name = os.path.join(label_path, label_name)
        if tmp_label_name[-4]!='.':
            continue
        else:
            label_num = label_num + 1
    
    # img num == label num
    if img_num != label_num:
        print('the num of img and label is not equl')
        num = img_num
    else:
        print('the num of img:', img_num)
        num = img_num
    
    # image enhancement for all images
    for i in range(int(num)):

        # i = random.randint(0, (num - 1))
        img_name = img_names[i]
        label_name = next((s for s in label_names if img_name[:-4] in s), None)
        if label_name == None:
            print(img_name,'没找到对应xml文件')
            continue
        tmp_img_name = os.path.join(img_path, img_name)
        tmp_label_name = os.path.join(label_path, label_name)

        # 读取文件并进行操作
        image = DataAugmentation.openImage(tmp_img_name)
        image.load()

        # 读取
        tree=ET.parse(tmp_label_name)
        label = tree.getroot()

        threadImage = [0] * 6
        _index = 0

        if len(opsList) != 1:
            for j in range(int(len(opsList))): # 多线程操作
                # j = random.randint(0, int(len(opsList) - 1))
                # j=4
                ops_name = opsList[j]
                threadImage[_index] = threading.Thread(target=imageOps,
                                                    args=(ops_name, image, label, new_img_path, new_label_path, img_name, label_name))
                threadImage[_index].start()
                _index += 1
                time.sleep(0.2)
                print({i}, {num})
        else:
            for ops_name in opsList: # 多线程操作
                threadImage[_index] = threading.Thread(target=imageOps,
                                                    args=(ops_name, image, label, new_img_path, new_label_path, img_name, label_name))
                threadImage[_index].start()
                _index += 1
                time.sleep(0.2)
                print({i}, {num})




if __name__ == '__main__':
    '''
    四个参数：原始图像输入路径, 增强图像输出路径
             原始标签输入路径, 增强标签输出路径
    '''
    EnhancedImage("E:\\photo\\00\\ImageSets\\",
              "E:\\photo\\00\\enhance_image\\",
              "E:\\photo\\00\\ImageSets\\Annotations/",
              "E:\\photo\\00\\enhance_image\\Annotations/")