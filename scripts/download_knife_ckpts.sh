# ckpts for knife task in real world
curl -L https://uofi.box.com/shared/static/akojvx5indl2fa81helkekcknnqz6zg2 --output latest.ckpt
mkdir -p data/outputs/knife/checkpoints
mv latest.ckpt data/outputs/knife/checkpoints
