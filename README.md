
# A Real-Time Human Action Recognition Model for Assisted Living

This project is developed using MMAction2 Framework to predict Falls, Staggering and Chest Pain from normal daily activities. It proposes a real-time HAR model combining a deep learning model (implemented with PyTorch) and a live video prediction and alert system to predict falls, staggering and chest pain from daily scenarios. Six advanced models were trained on a GPU using transfer learning, including transformer (TimeSFormer), 3D convolutional (I3D, SlowFast), and hybrid (UniFormerV2) architectures. Results are presented with class-wise and macro performance metrics, inference efficiency, model complexity and computational costs. The optimal model achieved a macro F1 score of 95.33% with superior inference throughput, utilized in the design of a real-time HAR model architecture.

##	Dataset Creation
The NTU RGB+D Action Recognition Dataset was utilized for this study. It included 948 videos each from the 'Falling', 'Staggering', and 'Chest Pain' categories within the 'Medical Conditions' section. For the 'Normal Scenario' class, 80 videos were randomly selected from each of the 40 'Daily Actions' categories, resulting in a diverse set of 3,200 videos depicting various everyday activities.

The dataset was split into training, validation, and testing segments with proportions of 75%, 12.5%, and 12.5% respectively, ensuring consistent representation across all classes. Features were mapped to labels, and annotation files were created, detailing video paths and corresponding labels. All procedures were documented and executed in the 'customise_datasets.ipynb' notebook, located in 'mmaction2/data/normal_vs_3critical', which includes both videos and their annotations.

##	Code Structure
The code is organised in a modular approach. Its main components are outlined below.
- “config” folder holds all model configuration scripts. In each config file, the inherited based model, training, validation and test pipelines, data loaders, and model training settings, are defined. 
-	“mmaction2/configs/recognition/slowfast” holds the downloaded pretrained models and customised models for SlowFast. 
-	“mmaction2/configs/recognition/i3d”  contains models for I3D. 
-	“mmaction2/configs/recognition/uniformerv2” contains defined models for UniFormer.
-	 “mmaction2/configs/recognition/timesformer” holds all models for TimeSformer.
-	The model training script is located at “mmaction2/tools/train.py”, and the testing script is defined in “mmaction2/tools/test.py”
-	All training and testing logs, including model checkpoints, are saved in “mmaction2/work_dirs”. 
-	The script used for calculating metrics (except for FLOPs and Parameters) and ploting graphs is located at “mmaction2/1_Evaluation_Plots_and_Investigation/evaluation_metric_analysis.ipynb”. Plots and analysis results presented in the report are located at "mmaction2/2 Updated_Evaluation_Plots_and_Investigation", which evaluates the six models results. The folder of “mmaction2/1_Evaluation_Plots_and_Investigation” contains the evaluation of the first four trained models.
-	Parameter size and Flops were calculated by calling “mmaction2/tools/analysis_tools/get_flops.py”
-	The script for loading model checkpoints and making inference for entire test data is located at “mmaction2/ Inference_test_data.py”
-	The script for make inference based on a model config file, a checkpoint file and a video path is located at “mmaction2/Inference.py”. The same code for calling using command-line arguments is defined at “mmaction2/Inference2.py”


