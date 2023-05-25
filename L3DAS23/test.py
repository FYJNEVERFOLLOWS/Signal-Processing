import os
import pickle
import torch
import torch.nn as nn

# path = "/Work21/2021/wanghonglong/datasets/L3DAS23_processed_100/task1_predictors_train.pkl"
# with open(path, 'rb') as f:
#     files = pickle.load(f)
# print(len(files[0]))

# print(len((os.listdir("/CDShare3/L3DAS23/Task1/L3DAS23_Task1_train100/data"))))
# print(len((os.listdir("/CDShare3/L3DAS23/Task1/L3DAS23_Task1_train360/data"))))

# with open("/Work21/2021/wanghonglong/datasets/L3DAS23_sepprocessed_100/test/task1_test_path.pkl", 'rb') as f:
#     training_path = pickle.load(f)
# print(training_path[0])

# from pathlib import Path
# path = Path("/Work21/2021/wanghonglong/datasets/L3DAS23_sepprocessed_100/test/task1_test_path.pkl")
# print(path.parent.absolute())

# with open("/Work21/2021/wanghonglong/datasets/L3DAS23_sepprocessed_100/test/84-121123-0008.pkl", 'rb') as f:
#     predictors = pickle.load(f)
# print(predictors[0])
# print(predictors[1])

# a = [1,2,3]
# b = [4,5,6]
# c = a+b
# print(c)

# a = torch.randn(1, 1, 24000)
# a = a.squeeze()
# print(a.shape)

a = torch.randn(2, 64000)
b = torch.randn(2, 64000)
ua = a.unsqueeze(-2)
ub = b.unsqueeze(-2)
loss_func = nn.MSELoss()
loss = loss_func(a, b)
lossu = loss_func(ua, ub)
print("loss:{}, lossu:{}".format(loss, lossu))

