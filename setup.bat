:: filepath: /workspace/STT_ML3/setup.bat
@echo off
:: Create and activate virtual environment
python -m venv venv
call venv\Scripts\activate

:: Upgrade pip and install dependencies
pip install --upgrade pip
pip install --upgrade tensorflow tensorflow_io speechbrain torchaudio librosa soundfile matplotlib

:: Create dataset directory and download datasets
mkdir ds
cd ds

curl -O https://www.openslr.org/resources/12/train-clean-100.tar.gz
curl -O https://www.openslr.org/resources/12/dev-clean.tar.gz
curl -O https://www.openslr.org/resources/12/test-clean.tar.gz

tar -xvzf train-clean-100.tar.gz
tar -xvzf dev-clean.tar.gz
tar -xvzf test-clean.tar.gz

cd ..

:: Deactivate virtual environment
deactivate