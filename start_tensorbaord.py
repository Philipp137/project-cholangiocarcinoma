import os
import webbrowser

log_file_dir = './train_log/'
webbrowser.open('http://localhost:6006')
os.system('tensorboard --logdir=' + log_file_dir)
