import numpy as np
import glob
import pickle
import os
import datetime
from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt
import matplotlib
import netCDF4 as nc
import h5py
import gc
# HDF5_USE_FILE_LOCKING = False
# m = Basemap(projection='laea', resolution='l', lat_ts=36, lon_0=130, lat_0=36, width=2500000,
#             height=2500000)
            # ,
            # llcrnrlon=113,llcrnrlat=24,urcrnrlon=147,urcrnrlat=48)

# def draw_picture(dir, datestr, lat, lon, chl):
#     plt.figure(1, (5, 5),clear=True)
#  # 兰伯特方位等积投影
#     x, y = m(lon, lat)
#     m.pcolormesh(x, y, np.log10(chl), vmin=-3, vmax=1.3)
#     # m.fillcontinents('gray')
#     plt.savefig(dir + datestr + '.WebP', dpi=1000)
#     plt.close()

nptime = []
datestart = datetime.datetime.now()
time = '00'
# dir = 'E:\\photo\\04\\'
dir2 = 'E:\\'
files = glob.glob('E:\\00\\G*.nc')
for file in files:
    datenow1 = datetime.datetime.now()
    a = nc.Dataset(file)
    chl = a.groups['geophysical_data'].variables['chlor_a'][:]
    lon = a.groups['navigation_data'].variables['longitude'][:]
    lat = a.groups['navigation_data'].variables['latitude'][:]
    a.close()

    year = int(file.split('\\')[-1][1:5])
    days = int(file.split('\\')[-1][5:8])
    date = datetime.datetime(year, 1, 1) + datetime.timedelta(days=days - 1)
    datestr = date.strftime('%Y%m%d')
    new = nc.Dataset(dir2 + time + '/' + datestr + time + 'CHL.nc', 'w', format='NETCDF4')
    new.createDimension('longitude', 5567)
    new.createDimension('latitude', 5685)
    new.createVariable('lonl', 'f',('latitude', 'longitude'), compression='zlib')
    new.createVariable('latl', 'f',('latitude', 'longitude'), compression='zlib')
    new.createVariable('chl', 'f', ('latitude', 'longitude'), compression='zlib')

    new.variables['chl'][:] = chl

    new.variables['latl'][:]= lat
    new.variables['lonl'][:] = lon
    new.variables['chl'].units = 'mg m^-3'
    new.close()
    # draw_picture(dir, datestr, lat, lon, chl)
    os.remove(file)
    print(datestr)
    gc.collect()
    datenow2 = datetime.datetime.now()
    nptime.append(datenow2 - datenow1)
    print(str(date) + ' 匹配所用时间：', datenow2 - datenow1)
    print('每天平均用时：', np.mean(nptime))
    print('完成预计时间：', datestart + np.mean(nptime) * (len(files)))

# files=glob.glob('F:/G2021*.nc')
# for file in files:
#     i+=1
#     plt.figure(i,(5.685, 5.567))
#     a=nc.Dataset(file)
#     np.linspace
#     lon=a.groups['navigation_data'].variables['longitude'][:]
#     lat=a.groups['navigation_data'].variables['latitude'][:]
#     chl=a.groups['geophysical_data'].variables['chlor_a'][:]
#     plt.imshow(np.log10(chl))
#     glatmin=a.groups['navigation_data'].variables['latitude'][:].data.min()
#     glatmax=a.groups['navigation_data'].variables['latitude'][:].data.max()
#     glonmin=a.groups['navigation_data'].variables['longitude'][:].data.min()
#     glonmax=a.groups['navigation_data'].variables['longitude'][:].data.max()
#     # print(glonmin,glonmax,glatmin,glatmax)
#     # plt.savefig('C:\\Users\\Administrator\\AppData\\Roaming\\JetBrains\\PyCharmCE2022.2\\scratches\\图片/'+file[42:55],dpi=1000)
#     plt.xticks(np.arange(0,5685,1),np.linspace(111.3342,148.66893,5685))
#     plt.yticks(np.arange(0,5567,1),np.linspace(21.548248,48.222153,5567))
#
# # plt.imshow(np.log10(b))
# pass
