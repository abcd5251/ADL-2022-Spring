import gdown
url = "https://drive.google.com/u/0/uc?id=1lwtLZOx5jnhkCHiNkoHA2Z36Likz9wIv&export=download"
output = "data.zip"

gdown.download(url,output)


model_url = "https://drive.google.com/u/0/uc?id=1FvJRNQNfxHfYCKlQcGc9KVQ0-mBWCtP7&export=download"
output_model = "./model.ckpt"
gdown.download(model_url,output_model)
