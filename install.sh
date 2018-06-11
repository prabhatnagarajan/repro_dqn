#!/bin/bash
pip install pipenv
pipenv --python 2.7
pipenv install torch=='0.3.1'
pipenv install torchvision=='0.2.0'
pipenv install numpy=='1.13.1'
pipenv install opencv-python=='3.4.0.12'
pipenv install matplotlib=='1.5.1'
# Install arcade learning environment
git clone https://github.com/mgbellemare/Arcade-Learning-Environment.git
cd Arcade-Learning-Environment/
apt-get install libsdl1.2-dev libsdl-gfx1.2-dev libsdl-image1.2-dev cmake
mkdir build && cd build
cmake -DUSE_SDL=ON -DUSE_RLGLUE=OFF -DBUILD_EXAMPLES=ON ..
make -j 4
cd ..
pipenv run python setup.py install