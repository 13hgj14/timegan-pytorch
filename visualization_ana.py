from metrics.visualization import *
import pickle

path="/mnt/lustre/GPU3/home/hugaojie/output/timegan/stock/test/"
with open(f"{path}/train_data.pickle", "rb") as fb:
    ori_data = pickle.load(fb)
    # ori_data = ori_data.reshape(ori_data.shape[0]*ori_data.shape[1], ori_data.shape[2])
with open(f"{path}/fake_data.pickle", "rb") as fb:
    new_data = pickle.load(fb)
    # new_data = new_data.reshape(new_data.shape[0]*new_data.shape[1], new_data.shape[2])
# print(ori_data)
# print(new_data)
visualization(ori_data, new_data,'stock', 'pca')
visualization(ori_data, new_data,'stock', 'tsne')