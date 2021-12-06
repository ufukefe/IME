#Create and activate evaluation environment
conda env create -f environment.yml

#Download algorithms DFM, patch2pix, SuperPoint, SuperGlue (Note: For SuperPoint, SuperGlue's repository is used.)
git clone https://github.com/ufukefe/dfm Algorithms/DFM
git clone https://github.com/GrumpyZhou/patch2pix Algorithms/patch2pix
git clone https://github.com/magicleap/SuperGluePretrainedNetwork Algorithms/SuperPoint
git clone https://github.com/magicleap/SuperGluePretrainedNetwork Algorithms/SuperGlue

#Download pretrained networks for algorithms
wget -P Algorithms/patch2pix/pretrained https://vision.in.tum.de/webshare/u/zhouq/patch2pix/pretrained/patch2pix_pretrained.pth
wget -P Algorithms/patch2pix/pretrained https://vision.in.tum.de/webshare/u/zhouq/patch2pix/pretrained/ncn_ivd_5ep.pth

#Move utils to algorithms' folders
#Learning-based Algorithms
cp utils/Algorithm_Wrappers/DFM/python/algorithm_wrapper.py Algorithms/DFM/python
cp utils/Algorithm_Wrappers/DFM/python/algorithm_wrapper_util.py Algorithms/DFM/python

cp utils/Algorithm_Wrappers/patch2pix/algorithm_wrapper.py Algorithms/patch2pix
cp utils/Algorithm_Wrappers/patch2pix/algorithm_wrapper_util.py Algorithms/patch2pix

cp utils/Algorithm_Wrappers/SuperPoint/algorithm_wrapper.py Algorithms/SuperPoint
cp utils/Algorithm_Wrappers/SuperPoint/descriptors_sp.py Algorithms/SuperPoint
cp utils/Algorithm_Wrappers/SuperPoint/environment.yml Algorithms/SuperPoint
cd Algorithms/SuperPoint
cp match_pairs.py match_pairs_sp.py
sed -i -e '272,282d' match_pairs_sp.py
sed -i '271r descriptors_sp.py' match_pairs_sp.py
rm descriptors_sp.py
cd ..
cd ..

cp utils/Algorithm_Wrappers/SuperGlue/algorithm_wrapper.py Algorithms/SuperGlue
cp utils/Algorithm_Wrappers/SuperGlue/environment.yml Algorithms/SuperGlue

#Classical Algorithms
cp utils/Algorithm_Wrappers/sift/algorithm_wrapper.py Algorithms/sift
cp utils/Algorithm_Wrappers/sift/algorithm_wrapper_util.py Algorithms/sift
cp utils/Algorithm_Wrappers/sift/environment.yml Algorithms/sift

cp utils/Algorithm_Wrappers/surf/algorithm_wrapper.py Algorithms/surf
cp utils/Algorithm_Wrappers/surf/algorithm_wrapper_util.py Algorithms/surf
cp utils/Algorithm_Wrappers/surf/environment.yml Algorithms/surf

cp utils/Algorithm_Wrappers/orb/algorithm_wrapper.py Algorithms/orb
cp utils/Algorithm_Wrappers/orb/algorithm_wrapper_util.py Algorithms/orb
cp utils/Algorithm_Wrappers/orb/environment.yml Algorithms/orb

cp utils/Algorithm_Wrappers/kaze/algorithm_wrapper.py Algorithms/kaze
cp utils/Algorithm_Wrappers/kaze/algorithm_wrapper_util.py Algorithms/kaze
cp utils/Algorithm_Wrappers/kaze/environment.yml Algorithms/kaze

cp utils/Algorithm_Wrappers/akaze/algorithm_wrapper.py Algorithms/akaze
cp utils/Algorithm_Wrappers/akaze/algorithm_wrapper_util.py Algorithms/akaze
cp utils/Algorithm_Wrappers/akaze/environment.yml Algorithms/akaze

# Download HPatches dataset
wget -P Datasets http://icvl.ee.ic.ac.uk/vbalnt/hpatches/hpatches-sequences-release.tar.gz

#Move utils to datasets' folders
cp utils/Datasets/HPatches/hpatches_organizer.py Datasets

# Extract HPatches
cd Datasets
tar -xvzf hpatches-sequences-release.tar.gz &> /dev/null

# Remove the high-resolution sequences
cd hpatches-sequences-release
rm -rf i_contruction i_crownnight i_dc i_pencils i_whitebuilding v_artisans v_astronautis v_talent
cd ..

# Organize HPatches for evaluation structure
conda run -n im_eval python3 hpatches_organizer.py
rm *.tar.gz
rm -rf hpatches-sequences-release
rm -rf hpatches_organizer.py
cd ..
cp utils/Datasets/HPatches/eval_hpatches.py Datasets/hpatches
