# ckpts for knife task in real world
curl -L https://uofi.box.com/shared/static/lzz81vd7ieydmz0b0qbwq8tmidghibrt --output epoch=200.ckpt
mkdir -p data/outputs/hang_mug/checkpoints
mv epoch=200.ckpt data/outputs/hang_mug/checkpoints
