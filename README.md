# `T3D-UDA` Spatio-Temporal Unsupervised Domain Adaptation 3D Sequential Data


`T3D-UDA` The source code was borrow from our work **Teachers in concordance for pseudo-labeling of 3D sequential data**
![img│center](./image/st-uda.png)
Train a model in source domain (i.e., Semantic Kitti dataset), and then perform unsupervised domain adaptation to target domain (i.e., Valeo data).

## Results
ST-UDA test/inference results on Valeo data. each color indicates the object class category (all 23 class ranging form 0 - 22).
![img|center](./image/test_results.png)
Curb Segmentation/Detection: ST-UDA test/inference results on Valeo data. Red color indicates curb and blue indicates other objects. 
![img|center](./image/curb_detection.png)

## Installation

### Pip Installation
#### Requirements
- PyTorch >= 1.7
- [torch-scatter](https://github.com/rusty1s/pytorch_scatter)
- [spconv](https://github.com/traveller59/spconv) (tested with spconv==2.0 and cuda==11.0)

OR
### Docker running/Installation
please run the following command to build the docker
```
cd ST-UDA # cd into this repository
```
then run
```
bash build_docker.sh
```
wait until the docker build finishes, then run 
```
bash run_docker.sh
```

## Data Preparation
Since the model requires inputs as npy file with information such as lidar, poses, and labels (for training purpose)
please make sure if the docker has pypcd module installed, if not please install it using:
```
pip3 install --upgrade git+https://github.com/klintan/pypcd.git
```
If you have PCD format data, please place them inside the dataset/pcd. then navigate to the "tools/data_conversion" and run pcd2npy.py
```
cd tools/data_conversion
python3 pcd2npy.py
```
 
To change model prediction results (NPY files) to PCD format, please navigate to the "tools/data_conversion" and run npy2pcd.py
```
cd tools/data_conversion
python3 npy2pcd.py
```
In all cases if you want to change path to the files, please check the source code.


### Valeo
Please create your dataset in the folder: "dataset", e.g:
```
dataset/valeo/...
```
The structure of the dataset should look like the directory tree bellow. Please also check the sample file inside the "dataset/valeo"

```
./	 
├── ...
└── dataset
    └── valeo # (...) sequences
         ├── X5-MIA2662_TIC_PCAP_20210616_065116_HIL_output/    
         │   ├── lidar/	
         │   │	├── 000000.npy
         │   │	├── 000001.npy
         │   │	└── ...
         │   ├── labels/  # optional (only used for training/evaluation)
         │   │   ├── 000000.npy
         │   │   ├── 000001.npy
         │   │   └── ...
         │   └── poses/
         │       ├── 000000.npy
         │       ├── 000001.npy
         │       └── ... 
         └── ...
```
### WOD
- Coming soon
```
./
├── 
├── ...
└── path_to_data_shown_in_config/
		├──Training
		│    └──sequences (...)
		├──Validation
		│    └──sequences (...)
		└──Testing
		     └──sequences (...)

```

## Testing/Inference
1. modify the configs/data_config/da_wod_valeo/da_wod_valeo_f3_3_32beam_time.yaml with your custom settings. We provide 
a sample yaml for valeo multi-frame (both past and future) aggregation
2. test the network by running 
   ```
   sh infer_wod_valeo_f3_3.sh
   ```
### Test/inference Results
The Test/inference results will be saved under the dataset directory (i.e., dataset/valeo/X5-MIA2662_TIC_PCAP_20210616_065116_HIL_output/) 
inside the folder predictions_f3_3; there will be class labels ranging form (0 - 22) within ".npy" files for each input frame. 
NB. class label "17" indicates curb, for more information regarding the class mapping, please check "valeo-multiscan.yaml"

### Evaluation
To evaluate the model performance, first please make sure there are labels provided for the data (check if "labels" directory with per-point labels for each frame exist under dataset/valeo/.../labels), then please run:
```
sh eval_wod_valeo_f3_3.sh
```

### Pretrained Models
-- Pretrained model for wod --> valeo found at
   - ./Model_checkpoint/da_wod_valeo_f3_3_beam32_time.pt
   
## TODO List
- [x] Provided Inference/test code for valeo dataset.
- [x] Support Future-frame supervision semantic segmentation.
- [x] Support Concordance of Teachers with Privilege Information.
- [X] Support Knowledge Distillation on single-frame and multi-frame semantic segmentation .
- [X] Release data preparation code.
- [ ] Release more pretrained model for wod --> valeo
- [ ] Integrate Teachers in Concordance for LiDAR 3D Object Detection into the codebase.
