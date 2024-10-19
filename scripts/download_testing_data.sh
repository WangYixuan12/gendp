# testing data for knife task in real world
mkdir -p data/outputs/knife/eval_0
curl -L https://uofi.box.com/shared/static/peod9daers8o8u2bh0osog2tiod7tqh9 --output episode_0.hdf5
mv episode_0.hdf5 data/outputs/knife/eval_0

mkdir -p data/outputs/knife/eval_1
curl -L https://uofi.box.com/shared/static/hflnvw6hb80vbc3ypsprm78uhp6gwlm7 --output episode_0.hdf5
mv episode_0.hdf5 data/outputs/knife/eval_1

