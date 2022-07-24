## Guitar Transcription w/ Inhibition
Code for the paper [A Data-Driven Methodology for Considering Feasibility and Pairwise Likelihood in Deep Learning Based Guitar Tablature Transcription Systems](https://arxiv.org/abs/2204.08094).
The repository contains scripts to do the following:
* Generate [JAMS](https://jams.readthedocs.io/en/stable/) files from [GuitarPro](https://www.guitar-pro.com/) data using [PyGuitarPro](https://pyguitarpro.readthedocs.io/en/stable/)
* Create a matrix of inhibition weights from a collection of tablature annotations (e.g., [DadaGP](https://github.com/dada-bots/dadaGP))
* Implements our proposed output layer formulation with inhibition
  * Interchangeable with [TabCNN](https://archives.ismir.net/ismir2019/paper/000033.pdf)'s output layer formulation
* Run six-fold cross-validation experiments on [GuitarSet](https://guitarset.weebly.com/)
* Train a (realtime capable) model for deployment
* Run inference (offline/online/realtime) on a specific track
* Run inference on audio from microphone
* and more...

The repository is built on top of [amt-tools](https://github.com/cwitkowitz/amt-tools), a more general music transcription repository.

## Installation
Clone the repository and install the requirements.
```
git clone https://github.com/cwitkowitz/guitar-transcription-with-inhibition
pip install -r guitar-transcription-with-inhibition/requirements.txt
```

(_Optional_) Install the repository for use within another project.
```
pip install -e guitar-transcription-with-inhibition/
```

## Usage
#### TODO
```
import guitar_transcription_inhibition
```

## Generated Files
The experiment root directory ```<root_dir>``` is one parameter defined at the top of the experiment script.
Execution of ```training/six_fold_experiment.py``` or ```training/train_to_deploy.py``` will generate the following under ```<root_dir>```:
 - ```n/```

    Folder (beginning at ```n = 1```) containing [sacred](https://sacred.readthedocs.io/en/stable/quickstart.html) experiment files:
 
     - ```config.json``` - parameter values for the experiment
     - ```cout.txt``` - contains any text printed to console
     - ```metrics.json``` - evaluation results for the experiment
     - ```run.json``` system and experiment information

    An additional folder (```n += 1```) with experiment files is created for each run where the name of the [sacred](https://sacred.readthedocs.io/en/stable/quickstart.html) experiment (```<root_dir>```) is the same. 

 - ```models/```

    Folder containing saved model and optimizer states at checkpoints, as well as the events files that tensorboard reads.

 - ```results/```

    Folder containing individual evaluation results for each track within the test set.

 - ```_sources/```

    Folder containing copies of the script(s) at the time(s) of execution.

Additionally, ground-truth will be saved under the path specified by ```features_gt_cache```, unless ```save_data=False```.

## Analysis
During training, losses and various validation metrics can be analyzed in real-time by running:
```
tensorboard --logdir=<root_dir>/models --port=<port>
```
Here we assume the current directory within the command-line interface contains ```<root_dir>```.
 ```<port>``` is an integer corresponding to an available port (```port = 6006``` if unspecified).

After running the command, navigate to <http://localhost:port> to view any reported training or validation observations within the tensorboard interface.

## Cite
##### SMC 2022 Paper
```
@inproceedings{cwitkowitz2022data,
  title     = {A Data-Driven Methodology for Considering Feasibility and Pairwise Likelihood in Deep Learning Based Guitar Tablature Transcription Systems},
  author    = {Frank Cwitkowitz and Jonathan Driedger and Zhiyao Duan},
  year      = 2022,
  booktitle = {Proceedings of Sound and Music Computing Conference (SMC)}
}
```
