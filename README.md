# PyTorch Template Project
A simple template project using PyTorch which can be modified to fit many deep learning projects.

## Basic Usage
The code in this repo is an MNIST example of the template, try run:
```
python main.py
```
The default arguments list is shown below:
```
usage: main.py [-h] [-b BATCH_SIZE] [-e EPOCHS] [--resume RESUME]
               [--verbosity VERBOSITY] [--save-dir SAVE_DIR]
               [--save-freq SAVE_FREQ] [--data-dir DATA_DIR]
               [--validation-split VALIDATION_SPLIT] [--no-cuda]

PyTorch Template

optional arguments:
  -h, --help    show this help message and exit
  -b BATCH_SIZE, --batch-size BATCH_SIZE
                        mini-batch size (default: 32)
  -e EPOCHS, --epochs EPOCHS
                        number of total epochs (default: 32)
  --resume RESUME
                        path to latest checkpoint (default: none)
  --verbosity VERBOSITY
                        verbosity, 0: quiet, 1: per epoch, 2: complete (default: 2)
  --save-dir SAVE_DIR
                        directory of saved model (default: model/saved)
  --save-freq SAVE_FREQ
                        training checkpoint frequency (default: 1)
  --train_log_step TRAIN_LOG_STEP
                        log frequency in number of iterations for training (default: 200)
  --val_log_step VAL_LOG_STEP
                        log frequency in number of iterations for validation (default: 2000)
  --data-dir DATA_DIR
                        directory of training/testing data (default: datasets)
  --validation-split VALIDATION_SPLIT
                        ratio of split validation data, [0.0, 1.0) (default: 0.0)
  --no-cuda   use CPU in case there's no GPU support
```

## Structure
```
├── base/ - abstract base classes
│   ├── base_data_loader.py - abstract base class for data loaders.
│   ├── etc.
│   └── base_trainer.py - abstract base class for trainers
│
├── data_loader/ - anything about data loading goes here
│   └── data_loader.py
│
├── data/ - dir containing saved models as well as log files
│
├── logger/ - for training process logging (generating Tensorboard readable files)
│   └── logger.py
│
├── model/ - models, losses, and metrics
│   ├── modules/ - submodules of your model
│       ├── loss.py
│       └── metric.py
│   └── model.py
|
├── test/ - test modules to make sure they work properly
│   └── test.py
│
├── trainer/ - trainers for your project
│   └── trainer.py
|
├── tester/ - tester for your project
│   └── tester.py
│
└── utils
     ├── utils.py
     └── ...

```

## Customization
### Loss/metrics
If you need to change the loss function or metrics, first ```import``` those function in ```main.py```, then modify this part:
```python
loss = my_loss
metrics = [my_metric]
```
The metrics and loss are going to be automatically added to the log file.

#### Multiple metrics
If you have multiple metrics in your project, just add it to the ```metrics``` list:
```python
loss = my_loss
metrics = [my_metric, my_metric2]
```
Now the logging shows two metrics.

## Tools
These are a set of functionality that can be used to improve the the execition times
by pre-processing the data or for some specific tasks to be done few times.

- **index_dataset**: this script generates an `index.h5` file containing a list
of all the files contained in the dataset directory. By loading this file we can
avoid to re-parse the entire set of files contained in the directory and speed up
the data loader class creation.
- **t_sne**: this script generates an `output.h5` file containing the data given
as input projected into a lower dimensional space (2 or 3) to be able to plot it and
visualize the distribution.
- **numerate_frames**: this script adds number to frames and it's very usefull when
we want to locate which is the corresponding frame we are visualizing in a video.

