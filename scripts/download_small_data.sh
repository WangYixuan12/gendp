curl -L https://uofi.box.com/shared/static/p670p4nsyqq84ythlbxiogs0qazcor70 --output small_data.zip
unzip small_data.zip
rm small_data.zip
mkdir -p data
mv small_data data/sapien_demo

