# Download models
mkdir settings && cd settings
gdown --id 1gzNFYMgYF8OA9_qnTYPlnwC4jJ2ZKKot
unzip model_retrieval.zip
rm model_retrieval.zip
rm Tfbm150E5-full-phobert-pyvi42/.zip

# Download preprocess data  
cd ..
gdown --id 19ddraetAxfzsq2eGNB71Si0DgllW8cjj
unzip preprocess_data.zip
rm preprocess_data.zip