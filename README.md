# CycleGAN-pytorch
slightly modified code based on [junyanz/pytorch-CycleGAN-and-pix2pix](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix)

dataset: horse2zebra/sketch2Line(our private dataset)

# TODO
- [ ] seperate options for cyclegan and ugatit

# Command Line

## Installation
`pip install -r requirements.txt`

## CycleGAN train/test
* To view training results and loss plots, run `python -m visdom.server` and click the URL [http://localhost:8097](http://localhost:8097).
* Train the model
`python train.py --data_root ./horse2zebra --name h2z_cyclegan --model cycle_gan`
* Test the model
`python test.py --data_root ./horse2zebra --name h2z_cyclegan --model cycle_gan`

* image preprocess
`python image_preprocess.py  --data_root ./rakugakiicon`

dataset: ./sketch2Line
`python train.py --data_root ./sketch2Line --name s2L_cyclegan --model cycle_gan`

UGATit
`python train.py --data_root ./dataset/sketch2Line --name s2L_UGATIT --model ugatit`
python train.py --data_root ./dataset/sketch2Line --name s2L_UGATIT --model ugatit --light True






