# Preprocessing from raw data
- The following preprocessing steps can be quite tedious. Please post issues if you cannot run the scripts.

- datasets: [Amazon](https://cseweb.ucsd.edu/~jmcauley/datasets/amazon/links.html)  
-- Rating file in `Files/Small subsets for experimentation`  
-- Meta files in `Per-category files`, [metadata], [image features]  


## Step by step
1. put the downloaded files in ./data/{dataset-name} folder
2. run 'python preprocessing.py -d {dataset-name}' for baseline and proprecess dataset and gen files
3. run 'python preprocessing-knockout-method.py -d {dataset-name} --method {e.g. noise}' for knockout
4. After preprocessing, you can use "python inspect_features.py -d {datasetName-method}" to see what the image and text feature look like 
