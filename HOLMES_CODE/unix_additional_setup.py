import sys
from setuptools import setup, Extension

# Cython compilation is not enabled by default

try:
    from Cython.Build import cythonize
except ImportError:
    use_cython = False
else:
    use_cython = True

on_mac = sys.platform.startswith('darwin')
on_windows = sys.platform.startswith('win')

MOD_NAME = 'brute_force_cython_ext'
MOD_PATH = 'duplib/search/brute_force_cython_ext'
COMPILE_LINK_ARGS = ['-O3', '-march=native', '-mtune=native']
# On Mac, use libc++ because Apple deprecated use of libstdc
COMPILE_ARGS_OSX = ['-stdlib=libc++']
LINK_ARGS_OSX = ['-lc++', '-nodefaultlibs']

ext_modules = []
if use_cython and on_mac:
    ext_modules += cythonize([
        Extension(
            MOD_NAME,
            [MOD_PATH + '.pyx'],
            language='c++',
            extra_compile_args=COMPILE_LINK_ARGS + COMPILE_ARGS_OSX,
            extra_link_args=COMPILE_LINK_ARGS + LINK_ARGS_OSX,
        )
    ], compiler_directives={'language_level' : "3"})
elif use_cython and on_windows:
    ext_modules += cythonize([
        Extension(
            MOD_NAME,
            [MOD_PATH + '.pyx'],
            language='c++',
        )
    ], compiler_directives={'language_level' : "3"})
elif use_cython:
    ext_modules += cythonize([
        Extension(
            MOD_NAME,
            [MOD_PATH + '.pyx'],
            language='c++',
            extra_compile_args=COMPILE_LINK_ARGS,
            extra_link_args=COMPILE_LINK_ARGS,
        )
    ], compiler_directives={'language_level' : "3"})
else:
    if on_mac:
        ext_modules += [Extension(MOD_NAME,
                                  [MOD_PATH + '.cpp'],
                                  extra_compile_args=COMPILE_ARGS_OSX,
                                  extra_link_args=LINK_ARGS_OSX,
                                  )
                        ]
    else:
        ext_modules += [Extension(MOD_NAME,
                                  [MOD_PATH + '.cpp'],
                                  )
                        ]
                        
setup(
    name='HOLMES',
    version='1.0.0',
    author='Anonymous',
    author_email='Anonymous',
    description='Package for HOLMES',
    long_description='Package for HOLMES',
    license='none',
    ext_modules=ext_modules
)
