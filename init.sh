if command -v python3 &>/dev/null; then
  python3 -m venv env
else
  echo "Python 3 is not installed. Please install Python 3 and try again."
  exit 1
fi
source env/bin/activate
which python
pip install --upgrade pip
pip install wheel setuptools
pip install -r requirements.txt