## Guitar Transcription with Inhibition
Code for the paper "[A Data-Driven Methodology for Considering Feasibility and Pairwise Likelihood in Deep Learning Based Guitar Tablature Transcription Systems](https://arxiv.org/abs/2204.08094)".
This repository contains scripts which do the following (and more):
* Acquire annotations in [JAMS](https://jams.readthedocs.io/en/stable/) format from [GuitarPro](https://www.guitar-pro.com/) data using [PyGuitarPro](https://pyguitarpro.readthedocs.io/en/stable/)
* Create a matrix of inhibition weights from a collection of tablature (e.g., [DadaGP](https://github.com/dada-bots/dadaGP))
* Implement the proposed tablature output layer formulation with inhibition
  * Interchangeable with the [TabCNN](https://archives.ismir.net/ismir2019/paper/000033.pdf) output layer formulation
* Perform six-fold cross-validation experiments on [GuitarSet](https://guitarset.weebly.com/)

The repository heavily utilizes [amt-tools](https://github.com/cwitkowitz/amt-tools), a more general music transcription repository.

## Installation
Clone the repository, install the requirements, then install the package:
```
git clone https://github.com/cwitkowitz/guitar-transcription-with-inhibition
pip install -r guitar-transcription-with-inhibition/requirements.txt
pip install -e guitar-transcription-with-inhibition/
```

## Usage
#### Convert GuitarPro to JAMS
Update ```<base_dir>``` (defined at the bottom of the script) to point to the top-level directory containing [GuitarPro](https://www.guitar-pro.com/) files, then run the following:
```
python guitar_transcription_inhibition/gpro/process_guitarpro.py
```
The corresponding [JAMS](https://jams.readthedocs.io/en/stable/) annotations will be placed under ```<base_dir>/jams```.

#### Computing Inhibition Weights
The process for computing inhibition weights for an arbitrary collection of symbolic tablature data involves extending the ```SymbolicTablature``` dataset wrapper and employing an ```InhibitionMatrixTrainer```.

Please see ```acquire_dadagp_matrix.py``` and ```acquire_guitarset_matrices.py``` under ```guitar_transcription_inhibition/inhibition``` for examples on how to compute inhibition weights for [DadaGP](https://github.com/dada-bots/dadaGP) and [GuitarSet](https://guitarset.weebly.com/), respectively.

#### Using Proposed Output Layer
The proposed tablature output layer can be initialized as follows:
```
from guitar_transcription_inhibition.models import LogisticTablatureEstimator

tablature_layer = LogisticTablatureEstimator(dim_in=<dim_in>,
                                             profile=<profile>,
                                             matrix_path=<matrix_path>,
                                             lmbda=<lmbda>)
```

Note that ```tablature_layer``` is an instance of ```torch.nn.Module```, and can be used as such.
It also acts as a standalone instance of ```amt_tools.models.TranscriptionModel```, and therefore implements the relevant functions ```TranscriptionModel.pre_proc``` and ```TranscriptionModel.post_proc```.

Please see the documentation in ```guitar_transcription_inhibition/models/tablature_layers.py``` and [amt-tools](https://github.com/cwitkowitz/amt-tools) for more information regarding the input arguments and usages within the [amt-tools](https://github.com/cwitkowitz/amt-tools) framework. 

#### Six-Fold Cross-Validation on GuitarSet
The scripts ```experiment.py``` and ```evaluation.py``` under ```six_fold_cv_scripts``` are also available as a more complete example of how to train and evaluate the proposed model under the six-fold cross-validation schema using [amt-tools](https://github.com/cwitkowitz/amt-tools).

## Generated Files
Execution of ```six_fold_cv_scripts/experiment.py``` will generate the following under ```<root_dir>``` (defined at the top of the script):
- ```n/``` - folder (beginning at ```n = 1```)<sup>1</sup> containing [sacred](https://sacred.readthedocs.io/en/stable/quickstart.html) experiment files:
  - ```config.json``` - parameter values used for the experiment
  - ```cout.txt``` - contains any text printed to console
  - ```metrics.json``` - evaluation results for the experiment
  - ```run.json``` - system and experiment information
- ```models/``` - folder containing saved model and optimizer state at each checkpoint, as well as an events file (for each execution) readable by [tensorboard](https://www.tensorflow.org/tensorboard)
- ```results/``` - folder containing separate evaluation results for each track within the test set
- ```_sources/``` - folder containing copies of scripts at the time(s) execution

Additionally, ground-truth and features will be saved under the path specified by ```gset_cache```, unless ```save_data=False```.
Scripts related to [GuitarPro](https://www.guitar-pro.com/) -- [JAMS](https://jams.readthedocs.io/en/stable/) conversion, inhibition matrix acquisition, visualization, and evaluation will also generate the respective files.

<sup>1</sup>An additional folder (```n += 1```) containing similar files is created for each execution with the same experiment name ```<EX_NAME>```.


## Analysis
During training, losses and various validation metrics can be analyzed in real-time by running:
```
tensorboard --logdir=<root_dir>/models --port=<port>
```
Here we assume the current working directory contains ```<root_dir>```, and ```<port>``` is an integer corresponding to an available port (```port = 6006``` if unspecified).

After running the above command, navigate to [http://localhost:&lt;port&gt;]() with an internet browser to view any reported training or validation observations within the tensorboard interface.

## Cite
##### SMC 2022 Paper ([Link](https://zenodo.org/record/6797681#.YtW0V7bMK8E))
```
@inproceedings{cwitkowitz2022data,
  title     = {A Data-Driven Methodology for Considering Feasibility and Pairwise Likelihood in Deep Learning Based Guitar Tablature Transcription Systems},
  author    = {Frank Cwitkowitz and Jonathan Driedger and Zhiyao Duan},
  year      = 2022,
  booktitle = {Proceedings of Sound and Music Computing Conference (SMC)}
}
```
