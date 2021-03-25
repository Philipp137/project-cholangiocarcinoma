import os
import webbrowser

log_file_dir = '/home/nb671233/project-cholangiocarcinoma/ml4medical/sbatch/lightning_logs/'
webbrowser.open('http://localhost:8008')
os.system('tensorboard --logdir=' + log_file_dir + ' --port=8008')
