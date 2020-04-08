# PyTorch Template Project
A simple template project using PyTorch which can be modified to fit many deep learning projects.

## Basic Usage
This repo contains train and test code. E.g.
```
python main.py
```
The default arguments list is shown below:
```
usage: train.py [-h] [-b BATCH_SIZE] [-e EPOCHS] [--resume RESUME]
                [--verbosity VERBOSITY] [--save-dir SAVE_DIR]
                [--save-freq SAVE_FREQ] [--data-dir DATA_DIR]
                [--validation-split VALIDATION_SPLIT] [--no-cuda]

PyTorch Template

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

