mkdir extra_packages
cd extra_packages

git clone https://github.com/ytopt-team/ConfigSpace.git
cd ConfigSpace
pip install -e .
cd ..

pip uninstall scikit-optimize
pip uninstall skopt

git clone https://github.com/ytopt-team/scikit-optimize.git
cd scikit-optimize
pip install -e .
cd ..

git clone -b version1 https://github.com/ytopt-team/autotune.git
cd autotune
pip install -e . 
cd ..

git clone -b main https://github.com/ytopt-team/ytopt.git
cd ytopt
pip install -e .
cd ..

sudo apt-get install autoconf automake libtool libgsl-dev

git clone git@github.com:argonne-lcf/CCS.git
cd CCS
./autogen.sh
mkdir build
cd build
../configure
make
make install
cd ../bindings/python
pip install parglare==0.12.0
pip install -e .
cd ../../..

pip install opentuner

sed -i "s@np.int)@np.int64)@g" scikit-optimize/skopt/space/transformers.py
