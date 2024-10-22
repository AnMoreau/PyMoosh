The scripts in this folder are designed as simple use cases that can be easily copy/pasted and modified.
They are very similar to the Jupyter Notebooks provided with the code, so have a look there if you need more details.

RUNNING CODE:

(For previous users) The context file is not necessary anymore!

For all users, if not using the last PyPi version (typically, because you want access to functions that are on github but not yet PyPi):
Before running install locally the library with
$ python3 -m pip install -e .
(from the command line, when in the root of the PyMoosh directory)
Also, now python doe not let you easily install external packages without
being in a python virtual environment.
To solve this, run the following once (creates a virtual env):
$ python -m venv /path/to/venv
And then when you want to use python (activates the virtual env):
$ source /path/to/venv/bin/activate
And then use pip as usual.