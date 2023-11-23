python3.9 -m venv env
source env/bin/activate
which python
pip install --upgrade pip
pip install wheel setuptools
pip install -r requirements.txt
PYTHONPATH=$PYTHONPATH:$(pwd)/env/lib/python3.9/site-packages
#PYOPENCL_CTX=''
export PYOPENCL_CTX='0'