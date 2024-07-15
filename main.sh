DIRNAME="$(dirname "$(readlink -f "$0")")"

OS_NAME=$(uname -s)
if [ "$OS_NAME" == "Darwin" ]; then
    PYTHON_PATH=bin
    PYTHON=python3
elif [ "$OS_NAME" == "Linux" ]; then
    PYTHON_PATH=bin
    PYTHON=python3
elif [ "${OS_NAME:0:5}" == "MINGW" ] || [ "${OS_NAME:0:7}" == "WINDOWS" ]; then
    PYTHON_PATH=Scripts
    PYTHON=python
else
    echo Unknown OS
    PYTHON_PATH=Scripts
    PYTHON=python
fi

if ! [ -d "$DIRNAME/.venv" ]; then
    echo Creating .venv...
    $PYTHON -m venv $DIRNAME/.venv
    $DIRNAME/.venv/$PYTHON_PATH/$PYTHON -m pip install --upgrade pip
    $DIRNAME/.venv/$PYTHON_PATH/pip install -r $DIRNAME/requirements.txt
fi

$DIRNAME/.venv/$PYTHON_PATH/$PYTHON $*

