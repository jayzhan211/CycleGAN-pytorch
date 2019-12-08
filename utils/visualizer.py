import numpy as np
import os
import sys
import ntpath
import time
from . import util, html
from subprocess import Popen, PIPE

if sys.version_info[0] == 2:
    VisdomExceptionBase = Exception
else:
    VisdomExceptionBase = ConnectionError


def save_images(webpage, visuals, image_path, aspect_ratio=1.0, image_size=256):
    """Save images to the disk.
    Parameters:
        webpage (the HTML class) -- the HTML webpage class that stores these imaegs (see html.py for more details)
        visuals (OrderedDict)    -- an ordered dictionary that stores (name, images (either tensor or numpy) ) pairs
        image_path (str)         -- the string is used to create image paths
        aspect_ratio (float)     -- the aspect ratio of saved images
        width (int)              -- the images will be resized to width x width
    This function will save images stored in 'visuals' to the HTML file specified by 'webpage'.
    """
    image_dir = webpage.get_image_dir()
    short_path = ntpath.basename(image_path[0])
    name = os.path.splitext(short_path)[0]

    webpage.add_header(name)
    ims, txts, links = [], [], []

    for label, im_data in visuals.items():
        im = util.tensor2im(im_data)
        image_name = '%s_%s.png' % (name, label)
        save_path = os.path.join(image_dir, image_name)
        util.save_image(im, save_path, aspect_ratio=aspect_ratio)
        ims.append(image_name)
        txts.append(label)
        links.append(image_name)
    webpage.add_images(ims, txts, links, image_size=image_size)


class Visualizer:
    def __init__(self, opt):
        """
        Initialize the Visualizer class
        :param opt: stores all the experiment flags; needs to be a subclass of BaseOptions
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
            self.nrow = opt.display_nrow
            self.vis = visdom.Visdom(server=opt.display_server, port=opt.display_port, env=opt.display_env)
            if not self.vis.check_connection():
                self.create_visdom_connections()

        """
        create an HTML object at <checkpoints_dir>/web/
        images will be saved under <checkpoints_dir>/web/images/
        """
        if self.use_html:
            self.web_dir = os.path.join(opt.checkpoints_dir, opt.name, 'web')
            self.img_dir = os.path.join(self.web_dir, 'images')
            print('create web directory {}...' .format(self.web_dir))
            util.mkdirs([self.web_dir, self.img_dir])

        # create a logging file to store training losses
        self.log_name = os.path.join(opt.checkpoints_dir, opt.name, 'loss_log.txt')
        with open(self.log_name, "a") as log_file:
            now = time.strftime("%c")
            log_file.write('================ Training Loss ({}) ================\n' .format(now))

    def reset(self):
        # Reset the self.saved status
        self.saved = False

    def print_current_losses(self, epoch, iters, losses, t_comp, t_data):
        """
        print current losses on console; also save the losses to the disk

        :param epoch: (int) current epoch
        :param iters: (int) current training iteration during this epoch (reset to 0 at the end of every epoch)
        :param losses: (OrderedDict) training losses stored in the format of (name, float) pairs
        :param t_comp: (float) computational time per data point (normalized by batch_size)
        :param t_data: (float) data loading time per data point (normalized by batch_size)
        :return:
        """
        message = '(epoch: {}, iters: {}, time: {:.3f}, data:{:.3f})'.format(epoch, iters, t_comp, t_data)
        for name, loss in losses.items():
            message += ', {}: {:.3f}'.format(name, loss)
        print(message)
        with open(self.log_name, "a") as log_file:
            log_file.write('{}\n'.format(message))

    def plot_current_losses(self, epoch, counter_ratio, losses):
        """
        display the current losses on visdom display: dictionary of error labels and values

        :param epoch: (int) current epoch
        :param counter_ratio: (float) progress (percentage) in the current epoch, between 0 to 1
        :param losses: (OrderedDict) training losses stored in the format of (name, float) pairs
        :return:
        """

        if not hasattr(self, 'plot_data'):
            self.plot_data = {'X': [], 'Y': [], 'legend': list(losses.keys())}
        self.plot_data['X'].append(epoch + counter_ratio)
        self.plot_data['Y'].append([losses[k] for k in self.plot_data['legend']])

        try:
            self.vis.line(
                X=np.stack([np.array(self.plot_data['X'])] * len(self.plot_data['legend']), 1),
                Y=np.array(self.plot_data['Y']),
                opts=dict(
                    title=self.name + 'loss over time',
                    legend=self.plot_data['legend'],
                    xlabel='epoch',
                    ylabel='loss',
                ),
                win=self.display_id,
            )
        except VisdomExceptionBase:
            self.create_visdom_connections()

    def create_visdom_connections(self):
        """
        If the program could not connect to Visdom server, this function will start a new server at port < self.port >
        """
        cmd = sys.executable + ' -m visdom.server -p {:d} &>/dev/null &' .format(self.port)
        print('\n\nCould not connect to Visdom server. \n Trying to start a server....')
        print('Command: {}'.format(cmd))
        Popen(cmd, shell=True, stdout=PIPE, stderr=PIPE)

    def display_current_results(self, visuals, epoch, save_result):
        """
        Display current results on visdom; save current results to an HTML file.

        :param visuals: (OrderedDict) dictionary of images to display or save
        :param epoch: (int) the current epoch
        :param save_result: (bool) if save the current results to an HTML file
        :return:
        """
        if self.display_id > 0:
            nrow = self.nrow
            if nrow > 0:
                nrow = min(nrow, len(visuals))
                h, w = next(iter(visuals.values())).shape[:2]
                table_css = """<style>
                            table {border-collapse: separate; border-spacing: 4px;
                            white-space: nowrap; text-align: center}
                            table td {width: % dpx; height: % dpx; padding: 4px; outline: 4px solid black}
                            </style>""" % (w, h)  # create a table css

                title = self.name
                label_html = ''
                label_html_row = ''
                images = []
                idx = 0
                img_np = None
                for label, image in visuals.items():
                    img_np = util.tensor2im(image)
                    label_html_row += '<td>{}</td>' .format(label)
                    images.append(img_np.transpose([2, 0, 1]))
                    idx += 1
                    if idx % nrow == 0:
                        label_html += '<tr>{}</tr>' .format(label_html_row)
                        label_html_row = ''
                if isinstance(img_np, np.ndarray):
                    white_image = np.ones_like(img_np.transpose([2, 0, 1])) * 255
                    while idx % nrow != 0:
                        images.append(white_image)
                        label_html_row += '<td></td>'
                        idx += 1
                    if label_html_row != '':
                        label_html += '<tr>{}</tr>'.format(label_html_row)

                try:
                    self.vis.images(images, nrow=nrow, win=self.display_id + 1,
                                    padding=2, opts=dict(title=title + ' images'))
                    label_html = '<table>{}</table>' .format(label_html)
                    self.vis.text(table_css + label_html, win=self.display_id + 2,
                                  opts=dict(title=title + ' labels'))
                except VisdomExceptionBase:
                    self.create_visdom_connections()
            else:
                idx = 1
                try:
                    for label, image in visuals.items():
                        image_numpy = util.tensor2im(image)
                        self.vis.image(image_numpy.transpose([2, 0, 1]), opts=dict(title=label),
                                       win=self.display_id + idx)
                        idx += 1
                except VisdomExceptionBase:
                    self.create_visdom_connections()

        if self.use_html and (save_result or not self.saved):
            self.saved = True
            # save images to the disk
            for label, image in visuals.items():
                image_numpy = util.tensor2im(image)
                img_path = os.path.join(self.img_dir, 'epoch{:03d}_{}.png' .format(epoch, label))
                util.save_image(image_numpy, img_path)

            # update website
            web_page = html.HTML(self.web_dir, 'Experiment name = {}' .format(self.name), refresh=0)
            for n in range(epoch, 0, -1):
                web_page.add_header('epoch [{}]'.format(n))
                ims, txts, links = [], [], []

                for label, _ in visuals.items():
                    img_path = 'epoch{:03d}_{}.png' .format(n, label)
                    ims.append(img_path)
                    txts.append(label)
                    links.append(img_path)
                web_page.add_images(ims, txts, links, width=self.win_size)
            web_page.save()


