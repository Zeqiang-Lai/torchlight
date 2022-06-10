import os

def filename(path):
    return os.path.splitext(os.path.basename(path))[0]

def fileext(path):
    return os.path.splitext(path)[1]

