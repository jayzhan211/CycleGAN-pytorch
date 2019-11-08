# CycleGAN-pytorch
slightly modified code based on [junyanz/pytorch-CycleGAN-and-pix2pix](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix)

dataset: horse2zebra/sketch2Line(our private dataset)

# TODO
- [ ] Add Colorization GAN

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
`python image_preprocess.py  --data_root ./dataset/rakugakiicon`

### dataset: ./sketch2Line

`python train.py --data_root ./dataset/sketch2Line --name s2L_cyclegan --model cycle_gan`

`python train.py --data_root ./dataset/sketch2Line --name s2L_cyclegan --model cycle_gan --continue`

### UGATit

`python train.py --data_root ./dataset/sketch2Line --name s2L_UGATIT --model ugatit`

`python train.py --data_root ./dataset/sketch2Line --name s2L_UGATIT --model ugatit --light True`

### unit_test.py

`python unit_test.py --data_root ./dataset/sketch2Line --name s2L_UGATIT --model cycle_gan`

`python unit_test.py --data_root ./dataset/sketch2Line --name s2L_cyclegan  --model cycle_gan --gray2rgb`

### CycleGAN_Coloorization

`python train.py --data_root ./dataset/sketch2Line --name s2L_CycleGAN_Colorization --moddel cycle_gan_colorization --input_nc 1 --dataset_mode colorization`


