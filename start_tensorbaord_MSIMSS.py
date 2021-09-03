import os
import webbrowser
from ml4medical.utils import get_project_root

root = get_project_root()
log_file_dir = root+'/MSIMSS/'
webbrowser.open('http://localhost:9009')
os.system('tensorboard --logdir=' + log_file_dir + ' --port=9009')
