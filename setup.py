from setuptools import setup
from Cython.Build import cythonize

#cython -a -3 --cplus NQTM-master/utils/preprocess.pyx

ext_options = {"compiler_directives": {"profile": True}, "annotate": True}

setup(
    name='preprocessing',
    ext_modules=cythonize(
        "C:\\Users\\mikad\\PycharmProjects\\MLP-Manosphere\\NQTM-master\\utils\\preprocess.pyx",
        compiler_directives={'language_level' : "3"}
    ),
    zip_safe=False,
)
