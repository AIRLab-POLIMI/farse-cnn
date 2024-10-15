# FARSE-CNN: Fully Asynchronous, Recurrent and Sparse Event-Based CNN
Official repository of our ECCV 2024 paper "FARSE-CNN: Fully Asynchronous, Recurrent and Sparse Event-Based CNN", by Riccardo Santambrogio, Marco Cannici and Matteo Matteucci.

## Installation
This code was developed and tested in a Docker container using the provided [Dockerfile](Dockerfile).
If you do not use Docker, make sure to install the necessary packages in your environment:
```
pip install -r requirements.txt
pip install torch-scatter --no-index -f https://data.pyg.org/whl/torch-1.13.0+cu116.html
```
Optionally, install Neptune for logging experiments.
```
pip install neptune
```

## Data Preparation

Please download (any of) the N-Cars, N-Caltech101 and Gen1 Automotive datasets from their official websites. The automatic download of DVS128 Gesture is supported by tonic.

All datasets should be placed under ```./data ```.

## Train
Training settings are controlled using ```.yaml``` configuration files. You can find our sample configurations in ```./configs```.
To start training, run the following script:
```
python train.py --train_cfg={path_to_cfg}
```

## Test
To test a model on object recognition or gesture recognition datasets, use the ```test.py``` script.
As with training, settings are controlled using ```.yaml``` configuration files.
```
python test.py --test_cfg={path_to_cfg}
```

To test object detection models, we use the tool provided in [Object-Detection-Metrics](https://github.com/rafaelpadilla/Object-Detection-Metrics).
You can generate the annotation files running:
```
python test_detect.py --test_cfg=configs/test_cfg_detect.yaml
```

## Citation
If you find this code useful, please cite:
```
@inproceedings{santambrogio2024farsecnn,
    author = {Santambrogio, Riccardo and Cannici, Marco and Matteucci, Matteo},
    title = {FARSE-CNN: Fully Asynchronous, Recurrent and Sparse Event-Based CNN},
    booktitle = {The European Conference on Computer Vision (ECCV)},
    year = {2024}
}
```