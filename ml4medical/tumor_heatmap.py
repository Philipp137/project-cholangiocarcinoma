import matplotlib.pyplot as plt
import glob
import os
import re
import numpy as np


def show_heatmap(xlist,ylist,predictions):
    """
    ** show_heatmap **
    :param xlist: xlist[i] is the x_i coordinate of the tile_i
    :param ylist: ylist[i] is the y_i coordinate of the tile_i
    :param predictions: predictions[i] is the prediction of the tile_i
    :return: figure handle
    """
    map = np.NaN * np.zeros([max(xlist) + 1, max(ylist) + 1])
    map[xlist, ylist] = predictions
    h = plt.imshow(map.T)
    plt.xticks([])
    plt.yticks([])
    return h


if __name__ =="__main__":
    project_directory = '/run/media/phil/Elements/data/CCC//14-51098/'
    tile_list = sorted(glob.glob(project_directory+'**/*.png', recursive=True))

    xlist , ylist, tile_num_list = [], [], []
    for tile_path in tile_list:
        file = os.path.basename(tile_path)
        m = re.match("(\d{2})-(\d+)_(\d+)_(\d+)_(\d+).[png]*",file)
        year = int(m.group(1))
        id = int(m.group(2))
        tile_number = int(m.group(3))
        x = int(m.group(4))
        y = int(m.group(5))
        print(f'tile: {file} year: 20{year} id: {id} (x,y) = {x,y}' )
        xlist.append(x)
        ylist.append(y)
        tile_num_list.append(tile_number)

    show_heatmap(xlist,ylist,predictions = tile_num_list)