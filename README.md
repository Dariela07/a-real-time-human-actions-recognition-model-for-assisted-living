
# A Real-Time Human Action Recognition Model for Assisted Living

This project is developed using MMAction2 Framework to predict health risks from normal daily activities. It proposes a real-time human action recognition (HAR) model combining a deep learning model (implemented with PyTorch) and a live video prediction and alert system to predict falls, staggering and chest pain from daily scenarios in assisted living environments. Six state-of-the-art models were trained on a GPU using transfer learning, including transformer (TimeSFormer), 3D convolutional (I3D, SlowFast), and hybrid (UniFormerV2) architectures. Results are presented with class-wise and macro performance metrics, inference efficiency, model complexity and computational costs. The optimal model achieved a macro F1 score of 95.33% with superior inference throughput, utilized in the design of a real-time HAR model architecture.

##	Dataset Creation
This study utilised the NTU RGB+D Action Recognition Dataset. All RGB video samples of Falling (948 videos), Staggering (948 videos), and Chest pain (948 videos) in the “Medical Conditions” category were selected to represent dangerous scenarios. The “Normal Scenario” class was formed by randomly selecting 80 videos from each of the 40 classes in the “Daily Actions” category to include a larger sample size (3,200 videos) with a wide diversity of daily activities, which simulate real-life conditions. 

The dataset was split into training, validation and testing sets, in proportion of 75%, 12.5% and 12.5% respectively. The splitting process ensured that the proportion of the four classes remained the same in training, testing and validation sets. Then, feature-label mapping was performed. An annotation text file for each set was created, listing the relative video path with its corresponding label. The dataset creation and label mapping were performed in “customise_datasets.ipynb”, which is located in the main folder. The created training, validation and testing datasets were saved in “mmaction2/data/normal_vs_3critical”, including both videos and the annotation test files. 

## MMAction2 Environment Setup

### 1. Create and Activate a Conda Environment
```bash
conda deactivate
conda create --name openmmlab_zz python=3.8 -y
conda activate openmmlab_zz
```

### 2. Install PyTorch and Dependencies
```bash
conda install pytorch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 pytorch-cuda=11.8 -c pytorch -c nvidia
```

### 3. Install Additional Dependencies
```bash
pip install fsspec
pip install -U openmim
```

### 4. Install OpenMMLab Frameworks
```bash
mim install mmengine
mim install mmcv==2.1.0
mim install mmdet==3.2.0  # Optional
mim install mmpose  # Optional
```

### 5. Clone and Install MMAction2
```bash
git clone https://github.com/open-mmlab/mmaction2.git
cd mmaction2
pip install -v -e .
```

## Testing the Framework

To verify the installation, run the following commands:

1. Download a pre-trained model:
```bash
mim download mmaction2 --config tsn_imagenet-pretrained-r50_8xb32-1x1x8-100e_kinetics400-rgb --dest .
```

2. Run a demo test:
```bash
python demo/demo.py tsn_imagenet-pretrained-r50_8xb32-1x1x8-100e_kinetics400-rgb.py \
    tsn_imagenet-pretrained-r50_8xb32-1x1x8-100e_kinetics400-rgb_20220906-2692d16c.pth \
    demo/demo.mp4 tools/data/kinetics/label_map_k400.txt
```

---

Following these steps ensures a smooth setup of the MMAction2 framework. If you encounter any issues, refer to the [official documentation](https://github.com/open-mmlab/mmaction2).


##	Code Structure
The code is organised in a modular approach. Its main components are outlined below.
- “config” folder holds all model configuration scripts. In each config file, the inherited based model, training, validation and test pipelines, data loaders, and model training settings, are defined. 
-	“mmaction2/configs/recognition/slowfast” holds the downloaded pretrained models and customised models for SlowFast. 
-	“mmaction2/configs/recognition/i3d”  contains models for I3D. 
-	“mmaction2/configs/recognition/uniformerv2” contains defined models for UniFormer.
-	 “mmaction2/configs/recognition/timesformer” holds all models for TimeSformer.
-	The model training script is located at “mmaction2/tools/train.py”, and the testing script is defined in “mmaction2/tools/test.py”
-	All training and testing logs, including model checkpoints, are saved in “<b>mmaction2/work_dirs</b>”. 
-	The script used for calculating metrics (except for FLOPs and Parameters) and ploting graphs is located at “mmaction2/1_Evaluation_Plots_and_Investigation/evaluation_metric_analysis.ipynb”. Plots and analysis results presented in the report are located at "<b>mmaction2/2 Updated_Evaluation_Plots_and_Investigation</b>", which evaluates the best performing variant of each model type from a total of six trained models. The folder of “mmaction2/1_Evaluation_Plots_and_Investigation” contains the evaluation of the first four trained models.
-	Parameter size and Flops were calculated by calling “mmaction2/tools/analysis_tools/get_flops.py”
-	The script for loading model checkpoints and making inference for entire test data is located at “mmaction2/ Inference_test_data.py”
-	The script for make inference based on a model config file, a checkpoint file and a video path is located at “mmaction2/Inference.py”. The same code for calling using command-line arguments is defined at “mmaction2/Inference2.py”


