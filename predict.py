# -----------------------------------------------------------------------#
#   predict.py将单张图片预测、摄像头检测、FPS测试和目录遍历检测等功能
#   整合到了一个py文件中，通过指定mode进行模式的修改。
# -----------------------------------------------------------------------#
import time
import json
import cv2
import numpy as np
from PIL import Image

from yolo import YOLO
from netCDF4 import Dataset

file_goci = 'img/2017071707CHL.nc'
data = Dataset(file_goci)
# points = np.column_stack((x.ravel(), y.ravel()))
lon = data.variables['lonl'][:]
lat = data.variables['latl'][:]
class_names=['AE','CE']
all_classes_nums=np.zeros(2)
def project_geo(point):
    point = np.array(point, dtype=int)
    return (float(lon[point[0], point[1]]), float(lat[point[0], point[1]]))


def project_daypicture(point, file):
    pix_point = []
    # 每个小图片的大小
    patch_size = yolo._defaults['input_shape'][0]

    # 重叠度
    eddy_radius = 100  # 要分辨的最大涡旋半径/km
    resolution = 500  # 图像分辨率/m
    overlap = eddy_radius * 1000 / resolution / patch_size  # 重叠度(最大半径/分辨率/patch大小)
    overlap_size = (int(patch_size * overlap), int(patch_size * overlap))
    row, col = int(file.split('_' or '.')[1]), int(file.split('_' or '.')[2][:-4])
    if row == str(int((lon.shape[1] - patch_size) / (patch_size - overlap_size[0])) + 1):
        upper = lon.shape[0] - patch_size
    else:
        upper = row * (patch_size - overlap_size[1])
    if col == str(int((lon.shape[0] - patch_size) / (patch_size - overlap_size[0])) + 1):
        left = lon.shape[1] - patch_size
    else:
        left = col * (patch_size - overlap_size[0])
    for i in range(point.shape[0]):
        for j in range(point.shape[1]):
            if j == 0 or j == 2:
                pix_point.append(int(point[i, j] + upper))
            elif j == 1 or j == 3:
                pix_point.append(int(point[i, j] + left))
            else:
                pix_point.append(point[i, j])
    pix_point = np.array(pix_point)
    return np.reshape(pix_point, (point.shape[0], point.shape[1]))


if __name__ == "__main__":
    yolo = YOLO()
    # ----------------------------------------------------------------------------------------------------------#
    #   mode用于指定测试的模式：
    #   'predict'           表示单张图片预测，如果想对预测过程进行修改，如保存图片，截取对象等，可以先看下方详细的注释
    #   'video'             表示视频检测，可调用摄像头或者视频进行检测，详情查看下方注释。
    #   'fps'               表示测试fps，使用的图片是img里面的street.jpg，详情查看下方注释。
    #   'dir_predict'       表示遍历文件夹进行检测并保存。默认遍历img文件夹，保存img_out文件夹，详情查看下方注释。
    #   'heatmap'           表示进行预测结果的热力图可视化，详情查看下方注释。
    #   'export_onnx'       表示将模型导出为onnx，需要pytorch1.7.1以上。
    # ----------------------------------------------------------------------------------------------------------#
    mode = "dir_predict"
    # -------------------------------------------------------------------------#
    #   crop                指定了是否在单张图片预测后对目标进行截取
    #   count               指定了是否进行目标的计数
    #   crop、count仅在mode='predict'时有效
    # -------------------------------------------------------------------------#
    crop = False
    count = True
    # ----------------------------------------------------------------------------------------------------------#
    #   video_path          用于指定视频的路径，当video_path=0时表示检测摄像头
    #                       想要检测视频，则设置如video_path = "xxx.mp4"即可，代表读取出根目录下的xxx.mp4文件。
    #   video_save_path     表示视频保存的路径，当video_save_path=""时表示不保存
    #                       想要保存视频，则设置如video_save_path = "yyy.mp4"即可，代表保存为根目录下的yyy.mp4文件。
    #   video_fps           用于保存的视频的fps
    #
    #   video_path、video_save_path和video_fps仅在mode='video'时有效
    #   保存视频时需要ctrl+c退出或者运行到最后一帧才会完成完整的保存步骤。
    # ----------------------------------------------------------------------------------------------------------#
    video_path = 0
    video_save_path = ""
    video_fps = 25.0
    # ----------------------------------------------------------------------------------------------------------#
    #   test_interval       用于指定测量fps的时候，图片检测的次数。理论上test_interval越大，fps越准确。
    #   fps_image_path      用于指定测试的fps图片
    #   
    #   test_interval和fps_image_path仅在mode='fps'有效
    # ----------------------------------------------------------------------------------------------------------#
    test_interval = 100
    fps_image_path = "img/street.jpg"
    # -------------------------------------------------------------------------#
    #   dir_origin_path     指定了用于检测的图片的文件夹路径
    #   dir_save_path       指定了检测完图片的保存路径
    #   dir_json            指定时间、类型、中心地理坐标、box 左上和右下点坐标、内接椭圆面积、可信度保存路径。
    #
    #   dir_origin_path和dir_save_path仅在mode='dir_predict'时有效
    # -------------------------------------------------------------------------#
    for str_hour in ['00','01','02','03','04','05','06','07']:
        hour = str_hour
        dir_origin_path = "E:\\photo\\clip\\" + hour + "\\"
        dir_save_path = "E:\\photo\\predict\\" + hour + "\\"
        dir_json = "E:\\photo\\predict\\" + hour + "\\" + "dataset\\"
        # -------------------------------------------------------------------------#
        #   heatmap_save_path   热力图的保存路径，默认保存在model_data下
        #
        #   heatmap_save_path仅在mode='heatmap'有效
        # -------------------------------------------------------------------------#
        heatmap_save_path = "model_data/heatmap_vision.png"
        # -------------------------------------------------------------------------#
        #   simplify            使用Simplify onnx
        #   onnx_save_path      指定了onnx的保存路径
        # -------------------------------------------------------------------------#
        simplify = True
        onnx_save_path = "model_data/models.onnx"

        if mode == "predict":
            '''
            1、如果想要进行检测完的图片的保存，利用r_image.save("img.jpg")即可保存，直接在predict.py里进行修改即可。 
            2、如果想要获得预测框的坐标，可以进入yolo.detect_image函数，在绘图部分读取top，left，bottom，right这四个值。
            3、如果想要利用预测框截取下目标，可以进入yolo.detect_image函数，在绘图部分利用获取到的top，left，bottom，right这四个值
            在原图上利用矩阵的方式进行截取。
            4、如果想要在预测图上写额外的字，比如检测到的特定目标的数量，可以进入yolo.detect_image函数，在绘图部分对predicted_class进行判断，
            比如判断if predicted_class == 'car': 即可判断当前目标是否为车，然后记录数量即可。利用draw.text即可写字。
            '''
            while True:
                img = input('Input image filename:')
                try:
                    image = Image.open(img)
                except:
                    print('Open Error! Try again!')
                    continue
                else:
                    r_image = yolo.detect_image(image, crop=crop, count=count)
                    r_image.show()

        elif mode == "video":
            capture = cv2.VideoCapture(video_path)
            if video_save_path != "":
                fourcc = cv2.VideoWriter_fourcc(*'XVID')
                size = (int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
                out = cv2.VideoWriter(video_save_path, fourcc, video_fps, size)

            ref, frame = capture.read()
            if not ref:
                raise ValueError("未能正确读取摄像头（视频），请注意是否正确安装摄像头（是否正确填写视频路径）。")

            fps = 0.0
            while (True):
                t1 = time.time()
                # 读取某一帧
                ref, frame = capture.read()
                if not ref:
                    break
                # 格式转变，BGRtoRGB
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                # 转变成Image
                frame = Image.fromarray(np.uint8(frame))
                # 进行检测
                frame = np.array(yolo.detect_image(frame))
                # RGBtoBGR满足opencv显示格式
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

                fps = (fps + (1. / (time.time() - t1))) / 2
                print("fps= %.2f" % (fps))
                frame = cv2.putText(frame, "fps= %.2f" % (fps), (0, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                cv2.imshow("video", frame)
                c = cv2.waitKey(1) & 0xff
                if video_save_path != "":
                    out.write(frame)

                if c == 27:
                    capture.release()
                    break

            print("Video Detection Done!")
            capture.release()
            if video_save_path != "":
                print("Save processed video to the path :" + video_save_path)
                out.release()
            cv2.destroyAllWindows()

        elif mode == "fps":
            img = Image.open(fps_image_path)
            tact_time = yolo.get_FPS(img, test_interval)
            print(str(tact_time) + ' seconds, ' + str(1 / tact_time) + 'FPS, @batch_size 1')

        elif mode == "dir_predict":
            import os
            from tqdm import tqdm
            import datetime
            import glob
            from torchvision.ops import nms
            import torch

            start_date = datetime.date(2011, 4, 1)
            end_date = datetime.date(2021, 3, 31)
            delta = datetime.timedelta(days=1)

            for current_date in tqdm(start_date + delta * n for n in range((end_date - start_date).days)):
                date_file = current_date.strftime('%Y%m%d')
                day_results = np.empty((0, 7))
                day_img_names = []
                img_names = glob.glob(dir_origin_path + date_file + '*.jpg')
                for img_name in img_names:
                    # img_name='20110404_8_8.jpg'
                    if img_name.lower().endswith(
                            ('.bmp', '.dib', '.png', '.jpg', '.jpeg', '.pbm', '.pgm', '.ppm', '.tif', '.tiff')):
                        image_path = os.path.join(dir_origin_path, img_name)
                        # image_path='E:\\photo\\00\\20110401.jpg'
                        image = Image.open(image_path)
                        img_name = img_name.split('\\')[-1]
                        r_image, results = yolo.detect_image(image, eddy_minradius=100)  # 直径50km
                        if results is None:
                            continue

                        day_results_pjt = project_daypicture(results, img_name)
                        day_results = np.concatenate((day_results, day_results_pjt), axis=0)
                        # day_img_names.append(img_name)
                        if not os.path.exists(dir_save_path):
                            os.makedirs(dir_save_path)
                        r_image.save(os.path.join(dir_save_path, img_name.replace(".jpg", ".png")), quality=95,
                                     subsampling=0)
                if day_results.shape[0] == 0:
                    continue
                day_results_t = torch.from_numpy(day_results)
                detections = day_results_t[:, :4]  # 获取坐标
                scores = day_results_t[:, 4] * day_results_t[:, 5]  # 获取置信度乘上分类概率
                nms_thres = yolo._defaults['nms_iou']  # 设置IoU阈值
                keep_idx = nms(detections, scores, nms_thres)
                day_results = day_results[keep_idx]
                classes_nums = np.zeros(2)
                for i in range(2):
                    if len(day_results.shape)==1:
                        day_results=np.reshape(day_results,(1,7))
                        num = np.sum(day_results[:,-1] == i)
                    else:
                        num = np.sum(day_results[:,-1] == i)
                    # if num > 0:
                    #     print(class_names[i], " : ", num, sep=' ')

                    classes_nums[i] = num

                date = img_name.split('_')[0]


                tempdict = {
                    "time": date,
                    "AE or CE label": "0 or 1",
                    "Anticyclonic": float((day_results[:, -1] == 0).sum()),
                    "Cyclonic": float((day_results[:, -1] == 1).sum()),
                    "results": {
                        "predict":  day_results.tolist(),
                        "eddy_type": list(int(day_results[i][-1]) for i in range(day_results.shape[0])),
                        "eddy_center_lon_lat": list(project_geo([(day_results[i, 2] + day_results[i, 0]) // 2,
                                                                 (day_results[i, 3] + day_results[i, 1]) // 2]
                                                                ) for i in range(day_results.shape[0])),
                        "box_min_lon_lat": list(
                            project_geo((day_results[i, 0], day_results[i, 1])) for i in
                            range(day_results.shape[0])),
                        "box_max_lon_lat": list(
                            project_geo((day_results[i, 2], day_results[i, 3])) for i in
                            range(day_results.shape[0])),
                        "eddy_inradius": list(float(min(day_results[i, 2] - day_results[i, 0],
                                                        day_results[i, 3] - day_results[i, 1]) * 500 / 2)
                                              for i in range(day_results.shape[0])),
                        "eddy_internal_ellipse_area": list(float((1 / 4 * np.pi * (
                                day_results[i, 2] - day_results[i, 0]) * 500 * 500 * (
                                                                          day_results[i, 3] -
                                                                          day_results[i, 1]))) for i in
                                                           range(day_results.shape[0])),
                        "confidence": list(float((day_results[i, 4] * day_results[i, 5])) for i in
                                           range(day_results.shape[0]))
                    }
                }

                if not os.path.isfile(os.path.join(dir_json, date_file + hour + '.json')):
                    with open(os.path.join(dir_json, date_file + '.json'), 'w') as f:
                        json.dump(tempdict, f)
                all_classes_nums+=classes_nums
                print(classes_nums, date + hour, all_classes_nums, sep=' ')
                # print(date_file+hour)
                # date=img_name.split('_')[0]
                # tempdict={
                #     "time": date,
                #     "AE":0,
                #     "CE":1,
                #     "img_name": {"predict_picture": day_img_names,
                #         "type": list(int(day_results[i][-1]) for i in range(day_results.shape[0])),
                #         "box_center_lon_lat": list(project([(day_results[i][2]+day_results[i][0])//2,(day_results[i][3]+day_results[i][1])//2],img_name) for i in range(day_results.shape[0])),
                #         "box_min_lon_lat": list(project((day_results[i][0],day_results[i][1]),img_name) for i in range(day_results.shape[0])),
                #         "box_max_lon_lat": list(project((day_results[i][2],day_results[i][3]),img_name )for i in range(day_results.shape[0])),
                #         "box_inradius":list(float(min(day_results[i][2]-day_results[i][0],day_results[i][3]-day_results[i][1])*500/2) for i in range(day_results.shape[0])),
                #         "box_internal_ellipse_area":list(float((1/4*np.pi*(day_results[i][2]-day_results[i][0])*500*500*(day_results[i][3]-day_results[i][1]))) for i in range(day_results.shape[0])),
                #         "confidence":list(float((day_results[i][4]*day_results[i][5])) for i in range(day_results.shape[0]))
                #     }
                # }
                #
                # if not os.path.isfile(os.path.join(dir_json,date+'.json')):
                #     with open(os.path.join(dir_json,date+'.json'), 'w') as f:
                #         json.dump(tempdict, f)
                # else:
                #     with open(os.path.join(dir_json,date+'.json'), 'r') as f:
                #         data = json.load(f)
                #     # 在原有内容的基础上添加新内容
                #     data.update({img_name[:-4]:tempdict[img_name[:-4]]})
                #     # 写入文件
                #     with open(os.path.join(dir_json,date+'.json'), 'w') as f:
                #         json.dump(data, f)
        elif mode == "heatmap":
            while True:
                img = input('Input image filename:')
                try:
                    image = Image.open(img)
                except:
                    print('Open Error! Try again!')
                    continue
                else:
                    yolo.detect_heatmap(image, heatmap_save_path)

        elif mode == "export_onnx":
            yolo.convert_to_onnx(simplify, onnx_save_path)

        else:
            raise AssertionError(
                "Please specify the correct mode: 'predict', 'video', 'fps', 'heatmap', 'export_onnx', 'dir_predict'.")
