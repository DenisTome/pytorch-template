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
│   ├── base_model.py - abstract base class for models.
│   └── base_trainer.py - abstract base class for trainers
│
├── data_loader/ - anything about data loading goes here
│   └── data_loader.py
│
├── data/ - dir containing saved models as well as log files
│
├── datasets/ - default dataset folder
│
├── logger/ - for training process logging (generating Tensorboard readable files)
│   └── logger.py
│
├── model/ - models, losses, and metrics
│   ├── modules/ - submodules of your model
│   ├── loss.py
│   ├── metric.py
│   └── model.py
│
├── trainer/ - trainers for your project
│   └── trainer.py
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

### Validation data
If you have separate validation data, try implement another data loader for validation, otherwise if you just want to split validation data from training data, try pass ```--validation-split 0.1```, in some cases you might need to modify ```utils/util.py```


## TODOs
- [x] Add support for multi-gpu training
- [x] Remove all ```print(.)``` instructions and subsitute them with ```logging```
- [x] Add Tensorboard support by generating files it can read; removing current logger
- [x] Add metric name used for exporting the metrics in the Tensorboard file as well
- [x] Add checkpoint saver during training (already defined in base_trainer)
- [x] Change train in base_trainer saving at the end of each epoch
- [x] Change ```save_freq``` in base_trainer for training every n iterations rather than epochs
- [ ] Add functionality to sample if needed the val set
- [x] Change ```_save_checkpoint``` in base_trainer taking as input both epoch and iteration and save file with right name
- [ ] Update ReadMe file
- [ ] Check that all the changes are working
- [ ] Add during training the graph definition
