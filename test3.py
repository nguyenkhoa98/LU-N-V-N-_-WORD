from imblearn.over_sampling import RandomOverSampler
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
# from mlxtend.preprocessing import TransactionEncoder
import time
import random
import statistics
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, recall_score
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import NearMiss
from sklearn.utils import resample
from imblearn.pipeline import make_pipeline
import random


data = pd.read_csv("data/data2.csv")
#   Lấy dữ liệu cho Demographic filter ---------------------------------------
#   Trích lọc lại dữ liệu data


#   Lấy dữ liệu GENDER, RFCM, TIME, INFLUENCE
data_info = data.iloc[:, [3, 7, 8, 20]]

#   Lấy dữ liệu sở thích
data_interest = data.iloc[:, 21:36]

#   Lấy dữ liệu về điểm số
data_score = data.iloc[:, 11:18]

#   Lấy dữ liệu về ngành học
data_major = data.iloc[:, 6]

#   Lấy dữ liệu dành cho đề tài này
data_project = pd.concat(
    [data_info, data_interest, data_score, data_major], axis=1)
data_project = data_project.dropna()
data_project = data_project.sample(frac=1)

#   Chia dữ liệu train 1
data_train = data_project.iloc[100:, :]
data_train_dm = data_train.iloc[:, :4]
data_train_ctb = data_train.iloc[:, 4:]

#   Chia dữ liệu test 1
data_test = data_project.iloc[:100, :]
data_test_dm = data_test.iloc[:, :4]
data_test_ctb = data_test.iloc[:, 4:]

list_tk = []
for i in range(len(data_project['Major'])):
    if data_project.iloc[i, 26] == 'Khoa học máy tính':
        list_tk.append('Nhóm ngành CNTT')
    elif data_project.iloc[i, 26] == 'Công nghệ thông tin':
        list_tk.append('Nhóm ngành CNTT')
    elif data_project.iloc[i, 26] == 'Truyền thông mạng máy tính':
        list_tk.append('Nhóm ngành CNTT')
    elif data_project.iloc[i, 26] == 'Tin học ứng dụng':
        list_tk.append('Nhóm ngành CNTT')
    elif data_project.iloc[i, 26] == 'Công nghệ phần mềm':
        list_tk.append('Nhóm ngành CNTT')
    elif data_project.iloc[i, 26] == 'Hệ thống thông tin':
        list_tk.append('Nhóm ngành CNTT')
    elif data_project.iloc[i, 26] == 'Quản trị kinh doanh':
        list_tk.append('Nhóm ngành kinh tế')
    elif data_project.iloc[i, 26] == 'Kế toán':
        list_tk.append('Nhóm ngành kinh tế')
    elif data_project.iloc[i, 26] == 'Marketing':
        list_tk.append('Nhóm ngành kinh tế')
    elif data_project.iloc[i, 26] == 'Kinh tế nông nghiệp':
        list_tk.append('Nhóm ngành kinh tế')
    elif data_project.iloc[i, 26] == 'Kinh doanh quốc tế':
        list_tk.append('Nhóm ngành kinh tế')
    elif data_project.iloc[i, 26] == 'Tài chính ngân hàng':
        list_tk.append('Nhóm ngành kinh tế')
    elif data_project.iloc[i, 26] == 'Kinh doanh thương mại':
        list_tk.append('Nhóm ngành kinh tế')
    elif data_project.iloc[i, 26] == 'Kiểm toán':
        list_tk.append('Nhóm ngành kinh tế')
    elif data_project.iloc[i, 26] == 'Sư phạm':
        list_tk.append('Nhóm ngành sư phạm')
    elif data_project.iloc[i, 26] == 'Sư phạm toán':
        list_tk.append('Nhóm ngành sư phạm')
    elif data_project.iloc[i, 26] == 'Sư phạm ngữ văn':
        list_tk.append('Nhóm ngành sư phạm')
    elif data_project.iloc[i, 26] == 'Sư phạm địa':
        list_tk.append('Nhóm ngành sư phạm')
    elif data_project.iloc[i, 26] == 'Giáo dục công dân':
        list_tk.append('Nhóm ngành sư phạm')
    elif data_project.iloc[i, 26] == 'Giáo dục thể chất':
        list_tk.append('Nhóm ngành sư phạm')
    elif data_project.iloc[i, 26] == 'Sư phạm hóa học':
        list_tk.append('Nhóm ngành sư phạm')
    elif data_project.iloc[i, 26] == 'Điều dưỡng':
        list_tk.append('Nhóm ngành y - dược')
    elif data_project.iloc[i, 26] == 'Y đa khoa':
        list_tk.append('Nhóm ngành y - dược')
    elif data_project.iloc[i, 26] == 'Răng hàm mặt':
        list_tk.append('Nhóm ngành y - dược')
    elif data_project.iloc[i, 26] == 'Dược sĩ':
        list_tk.append('Nhóm ngành y - dược')
    elif data_project.iloc[i, 26] == 'Y học dự phòng':
        list_tk.append('Nhóm ngành y - dược')
    elif data_project.iloc[i, 26] == 'Cao đẳng y tế':
        list_tk.append('Nhóm ngành y - dược')
    elif data_project.iloc[i, 26] == 'Y học cổ truyền':
        list_tk.append('Nhóm ngành y - dược')
    elif data_project.iloc[i, 26] == 'Thú y':
        list_tk.append('Nhóm ngành y - dược')
    elif data_project.iloc[i, 26] == 'Kỹ Thuật điện - điện tử':
        list_tk.append('Nhóm ngành thuộc khoa công nghệ')
    elif data_project.iloc[i, 26] == 'Vật Lý kĩ thuật':
        list_tk.append('Nhóm ngành thuộc khoa công nghệ')
    elif data_project.iloc[i, 26] == 'Cơ khí':
        list_tk.append('Nhóm ngành thuộc khoa công nghệ')
    elif data_project.iloc[i, 26] == 'Công nghệ thực phẩm':
        list_tk.append('Nhóm ngành thuộc khoa công nghệ')
    elif data_project.iloc[i, 26] == 'Quản lý công nghiệp':
        list_tk.append('Nhóm ngành thuộc khoa công nghệ')
    elif data_project.iloc[i, 26] == 'Điều Khiển tự động hóa':
        list_tk.append('Nhóm ngành thuộc khoa công nghệ')
    elif data_project.iloc[i, 26] == 'Công nghệ chế biến thủy sản':
        list_tk.append('Nhóm ngành thuộc khoa công nghệ')
    elif data_project.iloc[i, 26] == 'Luật':
        list_tk.append('Nhóm ngành thuộc khoa luật')
    elif data_project.iloc[i, 26] == 'Khoa học xã hội và nhân văn':
        list_tk.append('Nhóm ngành KHXH - NV')
    elif data_project.iloc[i, 26] == 'Văn học':
        list_tk.append('Nhóm ngành KHXH - NV')
    elif data_project.iloc[i, 26] == 'Thông tin học':
        list_tk.append('Nhóm ngành KHXH - NV')
    elif data_project.iloc[i, 26] == 'Triết học':
        list_tk.append('Nhóm ngành KHXH - NV')
    elif data_project.iloc[i, 26] == 'Phương đông học':
        list_tk.append('Nhóm ngành KHXH - NV')
    elif data_project.iloc[i, 26] == 'Việt Nam học':
        list_tk.append('Nhóm ngành KHXH - NV')
    elif data_project.iloc[i, 26] == 'Bảo vệ thực vật':
        list_tk.append('Nhóm ngành nông nghiệp')
    elif data_project.iloc[i, 26] == 'Nông học':
        list_tk.append('Nhóm ngành nông nghiệp')
    elif data_project.iloc[i, 26] == 'Nuôi trồng thủy sản':
        list_tk.append('Nhóm ngành nông nghiệp')
    elif data_project.iloc[i, 26] == 'Hóa học':
        list_tk.append('Nhóm ngành KHTN')
    elif data_project.iloc[i, 26] == 'Toán ứng dụng':
        list_tk.append('Nhóm ngành KHTN')
    elif data_project.iloc[i, 26] == 'Công nghệ sinh học':
        list_tk.append('Nhóm ngành KHTN')
    elif data_project.iloc[i, 26] == 'Sinh hóa':
        list_tk.append('Nhóm ngành KHTN')
    elif data_project.iloc[i, 26] == 'Sinh học ứng dụng':
        list_tk.append('Nhóm ngành KHTN')
    elif data_project.iloc[i, 26] == 'Quân đội':
        list_tk.append('Nhóm ngành QD - CAND')
    elif data_project.iloc[i, 26] == 'Hóa học':
        list_tk.append('Nhóm ngành KHTN')
    elif data_project.iloc[i, 26] == 'Toán ứng dụng':
        list_tk.append('Nhóm ngành KHTN')
    elif data_project.iloc[i, 26] == 'Công nghệ sinh học':
        list_tk.append('Nhóm ngành KHTN')
    elif data_project.iloc[i, 26] == 'Sinh hóa':
        list_tk.append('Nhóm ngành KHTN')
    elif data_project.iloc[i, 26] == 'Sinh học ứng dụng':
        list_tk.append('Nhóm ngành KHTN')
    elif data_project.iloc[i, 26] == 'Ngôn ngữ anh':
        list_tk.append('Nhóm ngành ngoại ngữ')
    elif data_project.iloc[i, 26] == 'Ngôn ngữ Pháp':
        list_tk.append('Nhóm ngành ngoại ngữ')
    elif data_project.iloc[i, 26] == 'Quản lý đất đai':
        list_tk.append('Nhóm ngành MTR & TNTN')
    elif data_project.iloc[i, 26] == 'Khoa học môi trường':
        list_tk.append('Nhóm ngành MTR & TNTN')
    elif data_project.iloc[i, 26] == 'Khoa học đất':
        list_tk.append('Nhóm ngành MTR & TNTN')
    elif data_project.iloc[i, 26] == 'Quản trị du lịch lữ hành':
        list_tk.append('Nhóm ngành du lịch & lữ hành')
    elif data_project.iloc[i, 26] == 'Quản trị khách sạn':
        list_tk.append('Nhóm ngành du lịch & lữ hành')
    else:
        list_tk.append('Nhóm ngành khác')

data_project['tk'] = list_tk


y2 = data_project.tk

# mang = ['Nhóm ngành CNTT','Nhóm ngành kinh tế','Nhóm ngành KHXH - NV',
# 'Nhóm ngành thuộc khoa công nghệ','Nhóm ngành nông nghiệp','Nhóm ngành y - dược',
# 'Nhóm ngành sư phạm','Nhóm ngành thuộc khoa luật','Nhóm ngành KHTN','Nhóm ngành QD - CAND',
# 'Nhóm ngành ngoại ngữ','Nhóm ngành MTR & TNTN','Nhóm ngành du lịch & lữ hành','Nhóm ngành khác'
# ]

# for i in range(len(data_project.tk)):
#     for j in range(len(mang)):
#         if data_project.iloc[i,27] == mang[j]:
#             data_project.iloc[i,27] = j

data_project[data_project['tk'] == 9]
data_project = data_project.drop(94)

y3 = data_project.tk
X2 = data_project.drop('tk', axis=1)
# X3 = X2.drop('Major', axis=1)

ros = RandomOverSampler(random_state=42)
X_resampled, y_resampled = ros.fit_resample(X2, y3)

data_end2 = pd.concat([X_resampled, y_resampled], axis=1)

data_sample0 = data_end2.iloc[:, :20]
data_sample = data_end2.iloc[:, 0:]

for i in range(data_sample.shape[0]):
    for j in range(data_sample.shape[1]-2):
        data_sample.iloc[i, j] = data_sample.iloc[i, j] + \
            random.uniform(-0.5, 0.5)

# data_sample = data_sample.drop('tk',axis = 1)

data_end5 = pd.concat([data_sample0, data_sample], axis=1)

# export_csv = data_plot.to_csv (r'C:\Users\KHUONG\Desktop\data_balance4.csv', index = None, header=True)
# y = data_end2.Major

# X1 = data_end2.drop('tk', axis=1)
# X = X1.drop('Major', axis=1)

# sm = SMOTE(random_state=42)
# X_res, y_res = sm.fit_resample(X, y)
# data_end3 = pd.concat([X_res,y_res],axis = 1)
# y = y.astype('int')

# X_res, y_res = pipe1.fit_resample(X, y)

# print('Resampled dataset shape %s' %Counter(y_res))

# data_end = pd.concat([X_res,y_res],axis = 1)

# data_end4 = data_end.sort_values('tk')

# k = df_upsampled.Major
# k = np.array(k)
# k = pd.DataFrame(k)


# data_end2 = pd.concat([data_end4,k],axis = 1)
# df_new = data_end2.rename(columns={'0': 'Major'})


# export_csv = data_end.to_csv (r'C:\Users\KHUONG\Desktop\data_resample3.csv', index = None, header=True)


# # Display new class counts
# df_upsampled.tk.value_counts()
# # export_csv = df_upsampled.to_csv (r'C:\Users\KHUONG\Desktop\data_balance2.csv', index = None, header=True)
# # data_project1 = data_project[data_project.tk == 'Nhóm ngành y - dược']
# # data_project1['Major'].drop_duplicates()

# data_resample_x = data_end.iloc[:,47]


# data_x0 = data_end5[data_end5['tk'] == 'Nhóm ngành CNTT']
# data_x1 = data_end5[data_end5['tk'] == 'Nhóm ngành KHXH - NV']
# data_plot = pd.concat([data_x0,data_x1],axis = 0)
# data_plot2 = data_plot.iloc[:,[39,43,47]]
# export_csv = data_plot2.to_csv (r'C:\Users\KHUONG\Desktop\data_plot_resampled.csv', index = None, header=True)

# X =data_plot2.iloc[:,[0,1]]
# y = data_plot2.iloc[:,2]

# plt.plot(X.iloc[0:177, 0], X.iloc[0:177, 1], 'bo')
# plt.plot(X.iloc[177:, 0], X.iloc[177:, 1], 'rx')
# plt.show()
