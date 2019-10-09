import ntpath
import os
import sys
from . import util, html
import time
import numpy as np
from subprocess import Popen, PIPE

if sys.version_info[0] == 2:
    VisdomExceptionBase = Exception
else:
    VisdomExceptionBase = ConnectionError


def save_images(web_page, visuals, image_path, aspect_ratio=0.1, image_size=256):
    """

    :param web_page: the HTML class,
    :param visuals: Ordered Dict that stores (name, images)
    :param image_path:
    :param aspect_ratio:
    :param image_size:
    :return:
    """
    image_dir = web_page.get_image_dir()
    short_path = ntpath.basename(image_path[0])
    name = os.path.splitext(short_path)[0]

    web_page.add_header(name)
    image_paths, images_name, links = [], [], []

    for label, img_data in visuals.items():
        img = util.tensor2img(img_data)
        image_name = '{}_{}.png' .format(name, label)
        save_path = os.path.join(image_dir, image_name)
        util.save_image(img, save_path, aspect_ratio=aspect_ratio)
        image_paths.append(image_name)
        images_name.append(label)
        links.append(image_name)
    web_page.add_images(image_paths, images_name, links, image_size=image_size)


class Visualizer:
    def __init__(self, opt):
        """

        :param opt:
        Step 1: Cache the training/test options
        Step 2: connect to a visdom server
        Step 3: create an HTML object for saveing HTML filters
        Step 4: create a logging file to store training losses
        """
        self.opt = opt
        self.display_id = opt.display_id
        self.use_html = opt.isTrain and not opt.no_html
        self.win_size = opt.display_window_size
        self.name = opt.name
        self.port = opt.display_port
        self.saved = False
        if self.display_id > 0:  # connect to a visdom server given <display_port> and <display_server>
            import visdom
            self.ncols = opt.display_ncols
            self.vis = visdom.Visdom(server=opt.display_server, port=opt.display_port, env=opt.display_env)
            if not self.vis.check_connection():
                self.create_visdom_connections()

        if self.use_html:  # create an HTML object at <checkpoints_dir>/web/; images will be saved under <checkpoints_dir>/web/images/
            self.web_dir = os.path.join(opt.checkpoints_dir, opt.name, 'web')
            self.img_dir = os.path.join(self.web_dir, 'images')
            print('create web directory {}...' .format(self.web_dir))
            util.mkdir([self.web_dir, self.img_dir])

        # create a logging file to store training losses
        self.log_name = os.path.join(opt.checkpoints_dir, opt.name, 'loss_log.txt')
        with open(self.log_name, "a") as log_file:
            now = time.strftime("%c")
            log_file.write('================ Training Loss ({}) ================\n' .format(now))

    def reset(self):
        self.saved = False

    def print_current_losses(self, epoch, iters, losses, t_comp, t_data):
        """

        :param epoch: (int) current epoch
        :param iters: (int) current iteration at this epoch
        :param losses: (OrderedDict) format(name, float)
        :param t_comp: (float) computational time per data point / batch_size
        :param t_data: (float) data loading time per data point / batch_size
        :return:
        """
        message = '(epoch: {}, iters: {}, time: {}, data:{})'.format(epoch, iters, t_comp, t_data)
        for name, loss in losses.item():
            message += '{}: {}'.format(name, loss)
        print(message)
        with open(self.log_name, "a") as log_file:
            log_file.write('{}\n'.format(message))

    def plot_current_losses(self, epoch, counter_ratio, losses):
        """

        :param epoch: current epoch
        :param counter_ratio: progress (percentage) in the current epoch, between 0 to 1
        :param losses: (name, float)
        :return:
        """
        if not hasattr(self, 'plot_data'):
            self.plot_data = {'X': [], 'Y': [], 'legend': list(losses.key())}
        self.plot_data['X'].append(epoch + counter_ratio)
        self.plot_data['Y'].append([losses[k] for k in self.plot_data['legend']])
        try:
            self.vis.line(
                X=np.stack([np.array(self.plot_data['X'])] * len(self.plot_data['legend']), 1),
                Y=np.array(self.plot_data['Y']),
                opts={
                    'title': self.name + 'loss over time',
                    'legend': self.plot_data['legend'],
                    'xlabel': 'epoch',
                    'ylabel': 'loss'
                },
                win=self.display_id
            )
        except VisdomExceptionBase:
            self.create_visdom_connections()

    def create_visdom_connections(self):
        """
        It start a new server at port < self.port >
        """
        cmd = sys.executable + ' -m visdom.server -p {} &>/dev/null &' .format(self.port)
        print('Could not connect to Visdom server.\nTrying to start a server....')
        print('Command: {}'.format(cmd))
        Popen(cmd, shell=True, stdout=PIPE, stderr=PIPE)





