import os
import time
from pathlib import Path

import torch
import torch_fidelity

import util.util as util
from data import create_dataset
from models import create_model
from options.train_options import TrainOptions
from translate import (create_train_dataset, create_val_dataset, save_images,
                       select_visuals)
from util.visualizer import Visualizer

if __name__ == '__main__':
  opt = TrainOptions().parse()   # get training options
  train_dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
  train_dataset_without_augmentations = create_train_dataset(opt)
  val_dataset = create_val_dataset(opt)
  train_dataset_size = len(train_dataset)    # get the number of images in the dataset.

  model = create_model(opt)      # create a model given opt.model and other options
  print('The number of training images = %d' % train_dataset_size)

  visualizer = Visualizer(opt)   # create a visualizer that display/save images and plots
  opt.visualizer = visualizer
  total_iters = 0                # the total number of training iterations

  optimize_time = 0.1
  smallest_val_fid = float('inf')
  times = []
  for epoch in range(opt.epoch_count, opt.n_epochs + opt.n_epochs_decay + 1):    # outer loop for different epochs; we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>
    epoch_start_time = time.time()  # timer for entire epoch
    iter_data_time = time.time()    # timer for data loading per iteration
    epoch_iter = 0                  # the number of training iterations in current epoch, reset to 0 every epoch
    visualizer.reset()              # reset the visualizer: make sure it saves the results to HTML at least once every epoch

    train_dataset.set_epoch(epoch)
    for i, data in enumerate(train_dataset):  # inner loop within one epoch
      iter_start_time = time.time()  # timer for computation per iteration
      if total_iters % opt.print_freq == 0:
        t_data = iter_start_time - iter_data_time

      batch_size = data["A"].size(0)
      total_iters += batch_size
      epoch_iter += batch_size
      if len(opt.gpu_ids) > 0:
        torch.cuda.synchronize()
      optimize_start_time = time.time()
      if epoch == opt.epoch_count and i == 0:
        model.data_dependent_initialize(data)
        model.setup(opt)               # regular setup: load and print networks; create schedulers
        model.parallelize()
      model.set_input(data)  # unpack data from dataset and apply preprocessing
      model.optimize_parameters()   # calculate loss functions, get gradients, update network weights
      if len(opt.gpu_ids) > 0:
        torch.cuda.synchronize()
      optimize_time = (time.time() - optimize_start_time) / batch_size * 0.005 + 0.995 * optimize_time

      if total_iters % opt.display_freq == 0:   # display images on visdom and save images to a HTML file
        save_result = total_iters % opt.update_html_freq == 0
        model.compute_visuals()
        visualizer.display_current_results(model.get_current_visuals(), epoch, save_result)

      if total_iters % opt.print_freq == 0:    # print training losses and save logging information to the disk
        losses = model.get_current_losses()
        visualizer.print_current_losses(epoch, epoch_iter, losses, optimize_time, t_data)
        if opt.display_id is None or opt.display_id > 0:
          visualizer.plot_current_losses(epoch, float(epoch_iter) / train_dataset_size, losses)

      if total_iters % opt.val_freq == 0:
        model.eval()

        model_translations_dir = Path(opt.checkpoints_dir, opt.name, 'translations')

        if not os.path.exists(model_train_translations_dir):
          os.mkdir(model_train_translations_dir)

        model_train_translations_dir = Path(model_translations_dir, 'train')
        model_val_translations_dir = Path(model_translations_dir, 'val')

        if not os.path.exists(model_train_translations_dir):
          os.mkdir(model_train_translations_dir)
        if not os.path.exists(model_val_translations_dir):
          os.mkdir(model_val_translations_dir)

        model_with_step_train_translations_dir = Path(
            model_train_translations_dir,
            'iter_%07d' % total_iters
        )
        model_with_step_val_translations_dir = Path(
            model_val_translations_dir,
            'iter_%07d' % total_iters
        )
        if not os.path.exists(model_with_step_train_translations_dir):
          os.mkdir(model_with_step_train_translations_dir)
        if not os.path.exists(model_with_step_val_translations_dir):
          os.mkdir(model_with_step_val_translations_dir)

        print('translating train...')
        for i, data in enumerate(train_dataset_without_augmentations):
          model.set_input(data)  # unpack data from data loader
          model.test()           # run inference
          visuals = select_visuals(model.get_current_visuals(), opt.direction)  # get image results
          img_path = model.get_image_paths()     # get image paths
          save_images(
              image_dir=model_with_step_train_translations_dir,
              visuals=visuals,
              image_path=img_path
          )
          if i % 5 == 0:
            print('processing (%04d)-th image... %s' % (i, img_path))

        print('translating val...')
        for i, data in enumerate(val_dataset):
          model.set_input(data)  # unpack data from data loader
          model.test()           # run inference
          visuals = select_visuals(model.get_current_visuals(), opt.direction)  # get image results
          img_path = model.get_image_paths()     # get image paths
          save_images(
              image_dir=model_with_step_val_translations_dir,
              visuals=visuals,
              image_path=img_path
          )
          if i % 5 == 0:
            print('processing (%04d)-th image... %s' % (i, img_path))

        target_real_train_dir = os.path.join(
            opt.dataroot,
            'trainB'
        )
        target_real_val_dir = os.path.join(
            opt.dataroot,
            'valB'
        )

        train_metrics = torch_fidelity.calculate_metrics(
            input1=str(target_real_train_dir),
            input2=str(Path(model_with_step_train_translations_dir)),  # fake dir,
            fid=True,
            verbose=False,
            cuda=torch.cuda.is_available(),
        )

        val_metrics = torch_fidelity.calculate_metrics(
            input1=str(target_real_val_dir),
            input2=str(Path(model_with_step_val_translations_dir)),  # fake dir,
            fid=True,
            verbose=False,
            cuda=torch.cuda.is_available(),
        )

        train_log_file = os.path.join(Path(opt.checkpoints_dir, opt.name), 'train_log.txt')
        val_log_file = os.path.join(Path(opt.checkpoints_dir, opt.name), 'val_log.txt')
        smallest_val_fid_file = os.path.join(Path(opt.checkpoints_dir, opt.name), 'smallest_val_fid.txt')

        with open(train_log_file, 'a') as tl:
          tl.write(f'step: {total_iters}\n')
          tl.write(
              f'frechet_inception_distance: {train_metrics["frechet_inception_distance"]}\n'
          )

        with open(val_log_file, 'a') as tl:
          tl.write(f'step: {total_iters}\n')
          tl.write(
              f'frechet_inception_distance: {val_metrics["frechet_inception_distance"]}\n'
          )

        if val_metrics['frechet_inception_distance'] < smallest_val_fid:
          smallest_val_fid = val_metrics['frechet_inception_distance']
          print('saving the smallest_val_fid model')
          model.save_networks('smallest_val_fid')

          if os.path.exists(smallest_val_fid_file):
            os.remove(smallest_val_fid_file)

          with open(smallest_val_fid_file, 'a') as tl:
            tl.write(
                f'step: {total_iters}\n'
            )
            tl.write(
                f'frechet_inception_distance: {val_metrics["frechet_inception_distance"]}\n'
            )

        model.train()

      if total_iters % opt.save_latest_freq == 0:   # cache our latest model every <save_latest_freq> iterations
        print('saving the latest model (epoch %d, total_iters %d)' % (epoch, total_iters))
        print(opt.name)  # it's useful to occasionally show the experiment name on console
        model.save_networks('latest')
        model.save_networks('iter_%07d' % total_iters)

      iter_data_time = time.time()

    if epoch % opt.save_epoch_freq == 0:              # cache our model every <save_epoch_freq> epochs
      print('saving the model at the end of epoch %d, iters %d' % (epoch, total_iters))
      model.save_networks('latest')
      model.save_networks(epoch)

    print('End of epoch %d / %d \t Time Taken: %d sec' % (epoch, opt.n_epochs + opt.n_epochs_decay, time.time() - epoch_start_time))
    model.update_learning_rate()                     # update learning rates at the end of every epoch.
