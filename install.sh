source conda activate
conda create -n erase python=3.8
conda activate erase 

pip install torch ==2.0.0
pip install torchvision==0.15.1+cu118
pip install torch-geometric==2.3.1
pip install matplotlib==3.7.1
pip install scikit-learn==1.3.0
pip install scipy==1.10.1
conda create -n rapids-23.4.1 -c rapidsai -c conda-forge -c nvidia cuml=23.4.1 python=3.8 cudatoolkit=11.8
pip install torch-scatter -f https://pytorch-geometric.com/whl/torch-2.0.0+cu118.html
pip install torch-sparse -f https://pytorch-geometric.com/whl/torch-2.0.0+cu118.html
pip install torch-cluster -f https://pytorch-geometric.com/whl/torch-2.0.0+cu118.html
pip install torch-spline-conv -f https://pytorch-geometric.com/whl/torch-2.0.0+cu118.html
pip install Ogb
