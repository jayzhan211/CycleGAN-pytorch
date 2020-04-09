# CycleGAN-pytorch
code based on [junyanz/pytorch-CycleGAN-and-pix2pix](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix)

```commandline
Ustyle V1 IN, out only, bilinear
Ustyle V1 ILN, out + x, nearest
```


# TODO
None

# Command Line

## Installation
`pip install -r requirements.txt`

## CycleGAN train/test
### To view training results and loss plots, run `python -m visdom.server` and click the URL [http://localhost:8097](http://localhost:8097).
*If you run without it is fine. The code will run the command automaitc*
### Train the model

`python train.py --dataroot ./dataset/selfie2anime --name selfie2anime_ustyle --model ustyle`

`python train.py --dataroot ./dataset/selfie2anime --name selfie2anime_ustyle --model ugatit --light`

#### Continue training

`python train.py --dataroot ./dataset/selfie2anime --name selfie2anime_ustyle --model ustyle --continue_train`

`python train.py --data_root ./horse2zebra --name h2z_cyclegan --model cycle_gan`

### Test the model

`python test.py --dataroot ./dataset/selfie2anime --name selfie2anime_ustyle --model ustyle`

`python test.py --data_root ./horse2zebra --name h2z_cyclegan --model cycle_gan`