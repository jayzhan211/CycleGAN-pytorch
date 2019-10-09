import os
from options.test_options import TestOptions
from data import create_dataset
from models import create_model
from utils import html
from utils.visualizer import save_images

if __name__ == '__main__':
    opt = TestOptions().parse()
    opt.num_threads = 0
    opt.batch_size = 1
    opt.display_id = -1  # no visdom display
    dataset = create_dataset(opt)
    model = create_model(opt)
    model.setup()

    # create a website
    web_dir = os.path.join(opt.results_dir, opt.name, '{}_{}'.format(opt.phase, opt.epoch))
    web_page = html.HTML(web_dir, 'Experiment = {}, Phase = {}, Epoch = {}' .format(opt.name, opt.phase, opt.epoch))
    if opt.eval:
        model.eval()
    for i, data in enumerate(dataset):
        if i == opt.num_test:
            break
        model.set_input(data)
        model.test()
        visuals = model.get_current_visuals()
        img_path = model.get_image_paths()
        if i % 5 == 0:
            print('processing ({:02d}) image... {}'.format(i, img_path))
            save_images(web_page, visuals, img_path,  aspect_ratio=opt.aspect_ratio, width=opt.display_winsize)




