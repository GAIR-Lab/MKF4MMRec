# MKF4MMRec (ECIR2026)
The source code for our Paper: ''Are Multimodal Embeddings Truly Beneficial for Recommendation? A Deep Dive into Whole vs. Individual Modalities''.

## How to run
1. Prepare dataset, download files from Amazon, put files in ./dataset
2. Use .py files in ./preprocess to preprocess dataset
3. python src/main.py -m {model_name} -d {dataset_name}-{method} to train model, such as "python src/main.py -m FREEDOM -d Baby-baseline" 