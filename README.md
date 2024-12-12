This project is developed using MMAction2 Framework to predict Falls, Staggering and Chest Pain from normal daily activities. 

3	Code Structure

The code is organised in a modular approach. Its main components are outlined below.
- “config” folder holds all model configuration scripts. In each config file, the inherited based model, training, validation and test pipelines, data loaders, and model training settings, are defined. 
-	“mmaction2/configs/recognition/slowfast” holds the downloaded pretrained models and customised models for SlowFast. 
•	“mmaction2/configs/recognition/i3d”  contains models for I3D. 
•	“mmaction2/configs/recognition/uniformerv2” contains defined models for UniFormer.
•	 “mmaction2/configs/recognition/timesformer” holds all models for TimeSformer.
•	The model training script is located at “mmaction2/tools/train.py”, and the testing script is defined in “mmaction2/tools/test.py”
•	All training and testing logs, including model checkpoints, are saved in “mmaction2/work_dirs”. 
•	The script used for calculating metrics (except for FLOPs and Parameters) and ploting graphs is located at “mmaction2/1_Evaluation_Plots_and_Investigation/evaluation_metric_analysis.ipynb”. Plots and analysis results presented in the report are located at the same folder “mmaction2/1_Evaluation_Plots_and_Investigation”
•	Parameter size and Flops were calculated by calling “mmaction2/tools/analysis_tools/get_flops.py”
•	The script for loading model checkpoints and making inference for entire test data is located at “mmaction2/ Inference_test_data.py”
•	The script for make inference based on a model config file, a checkpoint file and a video path is located at “mmaction2/Inference.py”. The same code for calling using command-line arguments is defined at “mmaction2/Inference2.py”
![Uploading image.png…]()

