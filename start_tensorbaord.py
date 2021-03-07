import os
import webbrowser

log_file_dir = './lightning_logs/'
webbrowser.open('http://localhost:6006')
os.system('tensorboard --logdir=' + log_file_dir)
