pip install --upgrade pip
pip install --upgrade tensorflow tensorflow_io  tensorflow-io speechbrain torchaudio librosa soundfile matplotlib


mkdir ds
cd ds

wget https://openslr.elda.org/resources/12/train-clean-100.tar.gz
wget https://openslr.elda.org/resources/12/dev-clean.tar.gz
wget https://openslr.elda.org/resources/12/test-clean.tar.gz

tar -xvzf dev-clean.tar.gz
tar -xvzf test-clean.tar.gz
tar -xvzf train-clean-100.tar.gz

