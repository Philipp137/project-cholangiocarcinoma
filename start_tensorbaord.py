import os
import webbrowser
from ml4medical.utils import get_project_root

root = get_project_root()
log_file_dir = root+'/ml4medical/sbatch/lightning_logs/'
webbrowser.open('http://localhost:8008')
os.system('tensorboard --logdir=' + log_file_dir + ' --port=8008')
