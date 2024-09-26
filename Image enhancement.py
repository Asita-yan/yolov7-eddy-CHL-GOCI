from PIL import Image, ImageFilter
from mpl_toolkits.basemap import Basemap
import logging
from datetime import datetime
from cv2 import filter2D
from netCDF4 import Dataset
from pint import UnitRegistry
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter
from scipy.signal import welch
from generic import (
    bbox_indice_regular,
    coordinates_to_local,
    distance,
    interp2d_geo,
    local_to_coordinates,
    nearest_grd_indice,
    uniform_resample,
)
from scipy.special import j1
from numpy import (
    arange,
    array,
    ceil,
    concatenate,
    cos,
    deg2rad,
    empty,
    errstate,
    exp,
    float_,
    floor,
    histogram2d,
    int_,
    interp,
    isnan,
    linspace,
    ma, log10, power,
)
from numpy import mean as np_mean
from numpy import (
    meshgrid,
    nan,
    nanmean,
    ones,
    percentile,
    pi,
    radians,
    round_,
    sin,
    sinc,
    where,
    zeros,
)
import glob
import numpy as np
from numba import njit
import matplotlib.pyplot as plt
import gc
import datetime
import cv2
from numba import jit

logger = logging.getLogger("pet")


# @njit(cache=True, fastmath=True)
def distancexy(x0, y0, x, y):
    """
    Compute distance between points from each line.

    :param float lon0:
    :param float lat0:
    :param float lon1:
    :param float lat1:
    :return: distance (in m)
    :rtype: array
    """

    sin_dlat = (x - x0)
    sin_dlon = (y - y0)

    a_val = sin_dlon ** 2 + sin_dlat ** 2
    return a_val


def finalize_kernel(kernel, order, half_x_pt, half_y_pt):
    # Symetry
    kernel_ = empty((half_x_pt * 2 * order + 1, half_y_pt * 2 * order + 1))
    kernel_[half_x_pt * order:] = kernel
    kernel_[: half_x_pt * order] = kernel[:0:-1]
    # remove unused row/column
    k_valid = kernel_ != 0
    x_valid = where(k_valid.sum(axis=1))[0]
    x_slice = slice(x_valid[0], x_valid[-1] + 1)
    y_valid = where(k_valid.sum(axis=0))[0]
    y_slice = slice(y_valid[0], y_valid[-1] + 1)
    return kernel_[x_slice, y_slice]


def estimate_kernel_shape(step_x_km, step_y_km, wave_length, order):
    # half size will be multiply with by order
    half_x_pt, half_y_pt = (
        ceil(wave_length / step_x_km).astype(int),
        ceil(wave_length / step_y_km).astype(int),
    )
    # x size is not good over 60 degrees
    y = arange(
        -step_y_km * half_y_pt * order,
        step_y_km * half_y_pt * order + 0.01 * step_y_km,
        step_y_km,
    )
    # We compute half + 1 and the other part will be compute by symetry
    x = arange(0,
               step_x_km * half_x_pt * order + 0.01 * step_x_km, step_x_km)
    y, x = meshgrid(y, x)
    dist_norm = distancexy(0, 0, x, y) / 1000.0 / wave_length
    return half_x_pt, half_y_pt, dist_norm


def kernel_lanczos(step_x_km, step_y_km, wave_length, order=1):
    """Not really operational
    wave_length in km
    order must be int
    """
    half_x_pt, half_y_pt, dist_norm = estimate_kernel_shape(
        step_x_km, step_y_km, wave_length, order,
    )
    kernel = sinc(dist_norm / order) * sinc(dist_norm)
    kernel[dist_norm > order] = 0
    return finalize_kernel(kernel, order, half_x_pt, half_y_pt)


def img_enhanced(img, add=20, cliplimit=6.0, tilegridsize=100):
    img_add = cv2.add(img, add, mask=np.uint8(~img.mask))
    # img_add[img_add>255]=255
    clahe = cv2.createCLAHE(clipLimit=cliplimit, tileGridSize=(tilegridsize, tilegridsize))
    img_enhanced = clahe.apply(img_add)
    img_enhanced = np.ma.array(img_enhanced, mask=img.mask)
    return img_enhanced


def float4color(zero2one):
    x = np.ma.multiply(np.ma.divide(zero2one - zero2one.min(), zero2one.max() - zero2one.min()), 256 ** 3)
    r = x % 256
    g = ((x - r) / 256 % 256)
    b = ((x - r - g * 256) / (256 * 256) % 256)
    # r = r.filled(255)
    # g = g.filled(255)
    # b = b.filled(255)
    r = r.astype('int')
    g = g.astype('int')
    b = b.astype('int')
    return r, g, b


def float1color(zero2one):
    x = np.ma.multiply(np.ma.divide(zero2one - zero2one.min(), zero2one.max() - zero2one.min()), 256)
    r = x % 256

    # r=r.filled(128)
    r = r.astype('int')
    # g = g.astype('int')
    # b = b.astype('int')
    return r


extend = False  # 不插值
circular = False  # 数据区域不是圆的不循环，不拓展边界
wavelength = 50
kernel = kernel_lanczos(0.5, 0.5, wavelength)
nptime = []
datestart = datetime.datetime.now()
num = 0
for year in range(2011, 2022):
    for i in range(1, 367):
        datenow1 = datetime.datetime.now()
        # if year!=2017 or i!=103:
        #     continue
        if len(str(i)) == 1:
            strtemp = '00' + str(i)
        elif len(str(i)) == 2:
            strtemp = '0' + str(i)
        else:
            strtemp = str(i)
        date = datetime.datetime(year, 1, 1) + datetime.timedelta(days=i - 1)
        datestr = date.strftime('%Y%m%d')
        # file_photo = glob.glob('E:\\photo\\04\\' + datestr + '.jpg')
        # if len(file_photo) == 1:
        #     continue
        files = glob.glob('f:\\07\\*' + str(year) + strtemp + '02*.L2_COMS_OC.nc')
        files.extend(glob.glob('f:\\07\\' + datestr + '*.nc'))
        if len(files) != 1:
            print(str(year) + strtemp + '不存在')
            continue

        dataset = Dataset(files[0])
        try:
            data = dataset.groups['geophysical_data'].variables['chlor_a'][:]
        except:
            data = dataset.variables['chl'][:]

        temp = (data.mask) | ((data > 20).data)
        # data[temp] = data.fill_value
        data = np.ma.array(data, mask=temp)
        # image = float4color(data)
        datamask = data.mask
        data = np.ma.log10(data)

        data_out = ma.empty(data.shape)
        data_out.mask = ones(data_out.shape, dtype=bool)

        d_lat = int((kernel.shape[1] - 1) / 2)
        d_lon = int((kernel.shape[0] - 1) / 2)

        tmp_matrix = ma.zeros((2 * d_lon + data.shape[0], 2 * d_lat + data.shape[1]))
        tmp_matrix.mask = ones(tmp_matrix.shape, dtype=bool)

        tmp_matrix[d_lon:-d_lon, d_lat:-d_lat] = data
        m = ~tmp_matrix.mask
        tmp_matrix[~m] = 0

        demi_x, demi_y = kernel.shape[0] // 2, kernel.shape[1] // 2
        values_sum = filter2D(tmp_matrix.data, -1, kernel)[demi_x:-demi_x, demi_y:-demi_y]
        kernel_sum = filter2D(m.astype(float), -1, kernel)[demi_x:-demi_x, demi_y:-demi_y]
        with errstate(invalid="ignore", divide="ignore"):
            if extend:
                data_out = ma.array(
                    values_sum / kernel_sum,
                    # mask=kernel_sum < (extend * kernel.sum()),
                    mask=kernel_sum == 0,
                )
            else:
                data_out = ma.array(
                    values_sum / kernel_sum,
                    # mask=kernel_sum < (extend * kernel.sum()),
                    mask=data.mask,
                )

        if extend:
            out = ma.array(data_out, mask=data_out.mask)
        else:
            out = ma.array(data_out, mask=data.mask + data_out.mask)
        if out.dtype != data.dtype:
            out.astype = (data.dtype)

        data_low_filter = data - out
        # data_low_filter = power(10,data) - power(10,out)
        # temp=np.ma.where((100 >= data_low_filter) & (data_low_filter> pow(10, 1.3)))
        # data_low_filter[temp]=power(10,1.3)
        # data_low_filter[np.ma.where(data_low_filter <0.001)] =0.001
        image1 = float1color(data_low_filter)

        image = np.ma.array(image1, dtype=np.uint8)
        add = 0
        clip = 255
        size = 100
        img = img_enhanced(image, add=int(add), cliplimit=clip, tilegridsize=size)
        # img = np.ma.array(img, mask=datamask)
        img = img.filled(np.uint8(0))
        cv2.imwrite('E:/photo/' + files[0].split('\\')[-1][8:10] + '/' + datestr + '.jpg', img)

        # image = float4color(data_low_filter)
        # image = np.array(image, dtype=np.uint8).transpose(1, 2, 0)
        # image = np.array(image).transpose(1, 2, 0)
        # pil_image = Image.fromarray(img)
        # pil_image = pil_image.filter(ImageFilter.DETAIL)
        # pil_image.save('E:/photo/' + files[0].split('\\')[-1][8:10] + '/' + datestr + '.jpg')

        gc.collect()
        datenow2 = datetime.datetime.now()
        nptime.append(datenow2 - datenow1)
        print(str(datestr) + ' 匹配所用时间：', datenow2 - datenow1)
        print('每天平均用时：', np.mean(nptime))
        print('完成预计时间：', datestart + np.mean(nptime) * (365 * 9 + 180))
