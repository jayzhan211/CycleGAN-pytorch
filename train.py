from options.train_options import TrainOptions
from data import create_dataset
from models import create_model
from utils.visualizer import Visualizer
import time

if __name__ == '__main__':
    opt = TrainOptions().parse()
    dataset = create_dataset(opt)
    dataset_size = len(dataset)
    model = create_model(opt)
    model.setup(opt)

    # model.update_learning_rate()
    # breakpoint()

    visualizer = Visualizer(opt)
    total_iters = 0
    t_data = 0
    for epoch in range(opt.epoch_count, opt.niter + opt.niter_decay + 1):
        print('Epoch_{} is starting'.format(epoch))
        # time for entire epoch
        epoch_start_time = time.time()
        # time of data loading
        iter_data_time = time.time()
        # the number of training iteration in cur epoch
        epoch_iter = 0

        for i, data in enumerate(dataset):
            iter_start_time = time.time()
            if total_iters % opt.print_freq == 0:
                t_data = iter_start_time - iter_data_time
            visualizer.reset()
            total_iters += opt.batch_size
            epoch_iter += opt.batch_size
            model.set_input(data)
            model.optimize_parameters()

            if total_iters % opt.display_freq == 0:
                save_result = total_iters % opt.update_html_freq == 0
                # model.compute_visuals()
                visualizer.display_current_results(model.get_current_visuals(), epoch, save_result)

            if total_iters % opt.print_freq == 0:
                losses = model.get_current_losses()
                t_comp = (time.time() - iter_start_time) / opt.batch_size
                visualizer.print_current_losses(epoch, epoch_iter, losses, t_comp, t_data)
                if opt.display_id > 0:
                    visualizer.plot_current_losses(epoch, float(epoch_iter) / dataset_size, losses)

            if total_iters % opt.save_latest_freq == 0:
                print('saving the latest model (epoch {}, total_iters {})'.format(epoch, total_iters))
                save_suffix = 'iter_{}'.format(total_iters) if opt.save_by_iter else 'latest'
                model.save_networks(save_suffix)

            iter_data_time = time.time()

        if epoch % opt.save_epoch_freq == 0:
            print('saving the model at epoch {}, iters {}'.format(epoch, total_iters))
            model.save_networks('latest')
            model.save_networks(epoch)

        print('End of epoch {} / {} \t Time: {} sec'.format(epoch,
                                                            opt.niter + opt.niter_decay,
                                                            time.time() - epoch_start_time))
        model.update_learning_rate()
        # print(model.optimizers[0].paramets)
