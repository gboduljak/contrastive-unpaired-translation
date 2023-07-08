import ntpath
import os
from pathlib import Path

import util.util as util
from data import CustomDatasetDataLoader
from models import create_model
from options.test_options import TestOptions


def create_train_dataset(opt):
    opt.phase='train'
    data_loader = CustomDatasetDataLoader(opt)
    dataset = data_loader.load_data()
    return dataset

def create_test_dataset(opt):
    opt.phase='test'
    data_loader = CustomDatasetDataLoader(opt)
    dataset = data_loader.load_data()
    return dataset

def select_visuals(visuals, direction):
  visual_key = 'fake_B' if direction == 'AtoB' else 'fake_A'
  visuals_to_exclude = [key for key in visuals.keys() if visual_key not in key]
  for visual in visuals_to_exclude:
    del visuals[visual]
  return visuals

def save_images(image_dir, visuals, image_path):
    short_path = ntpath.basename(image_path[0])
    name = os.path.splitext(short_path)[0]

    for label, im_data in visuals.items():
      im = util.tensor2im(im_data)
      image_name = '%s_%s.png' % (name, label)
      save_path = os.path.join(image_dir, image_name)
      util.save_image(im, save_path)


if __name__ == '__main__':
    opt = TestOptions().parse()  # get test options
    # hard-code some parameters for test
    opt.num_threads = 0   # test code only supports num_threads = 0
    opt.batch_size = 1    # test code only supports batch_size = 1
    opt.serial_batches = True  # disable data shuffling; comment this line if results on randomly chosen images are needed.
    opt.no_flip = True    # no flip; comment this line if results on flipped images are needed.
    opt.display_id = -1   # no visdom display; the test code saves the results to a HTML file.


    train_dataset = create_train_dataset(opt)  # create a dataset given opt.dataset_mode and other options
    test_dataset = create_test_dataset(opt)

    print(len(train_dataset))
    print(len(test_dataset))

    model = create_model(opt)      # create a model given opt.model and other options
    model.setup(opt)               # regular setup: load and print networks; create schedulers

    if opt.eval:
      model.eval()

    if not os.path.exists('translations'):
      os.mkdir('translations')

    model_translations_dir = Path('translations', opt.name)
    if not os.path.exists(model_translations_dir):
      os.mkdir(model_translations_dir)
    model_with_epochs_translations_dir = Path(model_translations_dir, f'epoch-{opt.epoch}')
    if not os.path.exists(model_with_epochs_translations_dir):
      os.mkdir(model_with_epochs_translations_dir)
        
    train_translated_imgs_dir = Path(model_with_epochs_translations_dir, 'train')
    test_translated_imgs_dir = Path(model_with_epochs_translations_dir, 'test')
    full_translated_imgs_dir = Path(model_with_epochs_translations_dir, 'full')

    if not os.path.exists(train_translated_imgs_dir):
      os.mkdir(train_translated_imgs_dir)
    if not os.path.exists(test_translated_imgs_dir):
      os.mkdir(test_translated_imgs_dir)
    if not os.path.exists(full_translated_imgs_dir):
      os.mkdir(full_translated_imgs_dir)


    print('processing train...')
    for i, data in enumerate(train_dataset):
      model.set_input(data)  # unpack data from data loader
      model.test()           # run inference
      visuals = select_visuals(model.get_current_visuals(), opt.direction)  # get image results
      img_path = model.get_image_paths()     # get image paths
      train_img_dir = Path(train_translated_imgs_dir, opt.direction)
      full_img_dir = Path(full_translated_imgs_dir, opt.direction)
      if not os.path.exists(train_img_dir):
         os.mkdir(train_img_dir)
      if not os.path.exists(full_img_dir):
         os.mkdir(full_img_dir)
      if i % 5 == 0:  # save images to an HTML file
          print('processing (%04d)-th image... %s' % (i, img_path))

      save_images(
        image_dir=train_img_dir,
        visuals=visuals,
        image_path=img_path
      )
      save_images(
        image_dir=full_img_dir,
        visuals=visuals,
        image_path=img_path
      )

    print('processing test...')
    for i, data in enumerate(test_dataset):
      model.set_input(data)  # unpack data from data loader
      model.test()           # run inference
      visuals = select_visuals(model.get_current_visuals(), opt.direction)  # get image results
      img_path = model.get_image_paths()     # get image paths
      test_img_dir = Path(test_translated_imgs_dir, opt.direction)
      full_img_dir = Path(full_translated_imgs_dir, opt.direction)
      if not os.path.exists(test_img_dir):
         os.mkdir(test_img_dir)
      if not os.path.exists(full_img_dir):
         os.mkdir(full_img_dir)
      if i % 5 == 0:  # save images to an HTML file
          print('processing (%04d)-th image... %s' % (i, img_path))

      save_images(
        image_dir=test_img_dir,
        visuals=visuals,
        image_path=img_path
      )
      save_images(
        image_dir=full_img_dir,
        visuals=visuals,
        image_path=img_path
      )
