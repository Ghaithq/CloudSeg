- To run the model on a certain directory of images please use this command:
```python "run_infernece.py" --dataset_path "/kaggle/input/cloud-masking-dataset/content/train/data" --model_path "model.pt"```
replace the paths with your paths, a csv file will be an output
- similarly for profiling
``` python profile.py model.pt 1 4 512 512```
- for detailed reporting please check the report pdf
