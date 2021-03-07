import shutil

def create_qupath_project_file(project_directory, filename,file):
    str = open('project_template.qpproj', 'r').read()
    text = str % (project_directory + "/project.qpproj", filename, file, filename, file)
    qp_proj_file = open(project_directory +"/project.qpproj", "w")
    n = qp_proj_file.write(text)
    qp_proj_file.close()
    return text, n

def create_qupath_classes_file(project_directory):
    shutil.copy('classes.json', project_directory +"/classifiers/")