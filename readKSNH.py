import pandas as pd
import numpy as np
from sklearn import preprocessing
# from sklearn.preprocessing import StandardScaler
dataset = pd.read_excel("KSNH.xlsx")
dataset = dataset.iloc[:,[0,1,2,3,4,6,7,8,9,10,11,12,13]] #Loại bỏ những cột Null

attribute_name = ['Gender', 'Interest', 'Major', 'RFCM','Time','Maths_score','Physics_score','Chemistry_score'
,'English_score','Literature_score','History_score','Geography_score','Influence']
dataset.columns = attribute_name


#Xử lý dữ liệu Giới Tính
def transform_encoder(feature_name, array_feature):
    le = preprocessing.LabelEncoder()
    le.fit(array_feature)
    dataset[feature_name] = le.transform(dataset[feature_name])
    return dataset

dataset = transform_encoder('Gender',['Nam','Nữ'])

#Xử lý dữ liệu Ngành học (Nhãn)
arr_major_names = [
['Khoa học máy tính','Khoa Học Máy Tính','Khoa Học máy tính','Khoa học máy tính ','Khoa học phân tích dữ liệu',' Khoa học máy tính','Khoa học dữ liệu','trí tuệ nhân tạo','An toàn thông tin']
,['Công nghệ thông tin','Công Nghệ Thông tin','Chuyên ngành Công Nghệ Thông Tin','Khoa học và kĩ thuật thông tin','Công Nghệ Thông Tin']
,['Truyền Thông và Mạng Máy tính','Truyền thông','TT & Mạng mt','Mạng máy tính và truyền thông','Kỹ thuật máy tính ','Truyền thông mạng máy tính']
,['Công nghệ phần mềm','Công nghệ phầm mềm','Kỹ thuật phần mềm'],['Tin học ứng dụng']
,['Hệ thống thông tin'],['Chính Trị Học','Chính trị học',"'\t' Chính trị học"]
,['Quản trị Kinh doanh','Quản trị Kinh doanh Tổng hợp ','Quản trị kinh doanh ','Quản trị kinh doanh','Kt','Thẩm định giá','Kinh tế','Kinh tế đối ngoại','Ngành Kinh Tế','Kinh tế đối ngoại ','Kinh tế học','Kinh Tế Quốc Tế ','Kinh tế quốc tế','Kinh tế Quốc tế','Kinh tế và quản lý công ','Kinh tế và quản lí công ','Quản trị du lịch và lữ hành ','Quản trị Kinh Doanh','Kinh Tế','Quản trị','quản trị kinh doanh','Quản trị kinh doanh','QTKD','Quản Trị Kinh Doanh','Quản trị kinh doanh tổng hợp','Quản trị kinh doanh vận tải hàng không']
,['Quản lý đất đai','Quàn lý đất đai'],['Kế Toán','Kế toán','kế toán','KẾ toán']
,['Luật Tư Pháp','Luật kinh tế','Luật Kinh tế','Luật thương mại','Luật Thương mại','Luật dân sự','Luật ','Luật tư pháp']
,['Quản lý tài nguyên – môi trường','Tài nguyên và Môi Trường','Kinh tế tài nguyên thiên nhiên',' Quản lý tài nguyên – môi trường','Kỹ thuật Môi trường','Công nghệ Kỹ thuật Môi trường','Công nghệ kỹ thuật Môi trường ','Công nghệ kỹ thuật Môi trường','Công nghệ Kĩ thuật Môi trường ','Khoa học môi trường','Quản lý tài nguyên và môi trường','Quản lý môi trường','Môi trường']
,['Xã Hội Và Nhân Văn','Xã hội học','Xã Hội Học','XHH','Xã hội học ','KHXHVNV','Khoa học xã hội và nhân văn']
,['Kỹ thuật công trình xây dựng','','Kỹ thuật xây dựng công trình giao thông','civil e','Kỹ thuật xây dựng','Kỹ Thuật xây dựng','Kỹ thuật Xây dựng','Xây dựng','Xây dựng dân dụng','Xây dựng cầu đường','Cnktct xây dựng','KTXD','Kỹ thuật công trình  xây dựng','Kỹ thuật xây Dựng ','KT Xây Dựng','Kỹ Thuật Xây Dựng','Kỹ thuật Xây Dựng ']
,['Điều dưỡng','Điều dưỡng '],['Dược học','Dược sĩ','Dược','Đại học dược','Dược sỹ','Dược ','Ngành Dược Học','Y dược']
,['Văn học','Quản lý văn hoá','Tiếng Việt và văn hóa Việt Nam','Văn học ','Truyền thông văn hoá ','Truyền thông văn hóa ']
,['Ngôn ngữ Anh','Ngôn Ngữ Anh','TIẾNG ANH THƯƠNG MẠI','TIẾNG ANH THƯƠNG MẠI ','Ngôn ngữ anh','Ngôn ngữ','Ngoại ngữ','Ngoại Ngữ','Sư phạm Tiếng Anh','ngoại ngữ-ngôn ngữ học','Ngôn ngữ ']
,['y khoa','GMHS18','Điều Dưỡng','Ngành Y','Y sĩ','Y đa khoa','Y Đa Khoa','Khoa Y','Bác sĩ','kỹ thuật xét nghiệm y học','xét nghiệm','Kỹ Thuật Xét Nghiệm Y Học','Kỹ thuật xét nghiệm y học', 'y','Xét nghiệm y học','Y','Y sĩ đa khoa','Y khoa','Y Sỹ Đa Khoa','Xét Nghiệm Y học']
,['y học dự phòng','Y học dự phòng','Y học sức khỏe','Sức khỏe','Y học','Y Sĩ Đa Khoa ','Y '],['thông tin học','Truyền thông đa phương tiện']
,['Bác sĩ Răng Hàm Mặt','Răng-hàm-mặt','Bác sĩ răng hàm mặt','Răng Hàm Mặt','Nha khoa','Răng hàm mặt','Bác sĩ Răng Hàm mặt']
,['Thiết kế đồ họa','Thiết kế đồ hoạ','Thiết kế đồ họa ','Công nghệ kỹ thuật hóa học, Mỹ thuật đa phương tiện','Thiết kế đồ họa 2D','Thiết Kế Đồ Hoạ','Thiế kế đồ hoạ','Mỹ Thuật Đa Phương Tiện','Thiết Kế Đồ Họa','MTCN','Mỹ thuật đa phương tiện','Thẩm mỹ','thiết kế đồ họa']
,['Công nghệ Kỹ thuật điện','Công nghệ Kỹ thuật điện, điện tử','Cồn nghệ kĩ thuật in','Công nghệ kỹ thuật in','điện tử','Diện-Điện Tử','Điện -điện tử','Điện  - Điện Tử','Điện - điện từ','Cơ điện tử','Ky thuật điện','Điện - điện tử','Điện tử','Kỹ thuật điện - điện tử','Hệ thống điện','Hệ thống điện ','Kỹ Thuật điện','Kĩ thuật điện','Điện Tử']
,['Công nghệ may'],['Sư phạm','Sư phạm ','Ngành giáo dục đặc biệt','Giáo viên'],['Cơ Khí Giao Thông','Cơ khí chế tạo máy','Kỹ Thuật Cơ Khí','Kỹ thuật cơ khí','Bảo trì sữa chửa oto','Công nghệ ÔTÔ','Công nghệ ÔTÔ ','Công nghệ oto','Công nghệ ôtô','Ô tô','ô tô','Kỹ thuật ô tô','Công nghệ oto ','Công Nghệ Ô Tô','Công nghệ ô tô','công nghệ kỹ thuật ô tô','Kỹ thuật oto','Công nghệ kỹ thuật ô tô','Công nghệ ÔTÔ','Công nghệ Ô tô']
,['Hóa học','Công nghệ kỹ thuật hóa học','Công nghệ kỹ thuật hoá học','Hóa dược','Kỹ Thuật Hoá Học'],['cao đẳng y tế ĐN']
,['quân đội'],['Bảo vệ thực vật'],['Công nghệ thực phẩm','Công nghệ Thực phẩm','Công nghệ thực phẩm '],['Công nghệ sau thu hoạch','Công nghệ rau hoa quả và cảnh quan','Công nghệ rau hoa quả và cảnh quan ','Khoa học cây trồng','Nông Học','Nông Học','Nông học']
,['Tài chính ngân hàng','Tài chính Ngân hàng','tài chính ngân hàng','TCNH','Tài chính ngân hàng ','Tài chính ','Tài chính - Ngân hàng','Ngân Hàng','Tài chính- Ngân hàng','Tài chính-ngân hàng','Tài Chính Ngân Hàng','Tài chính'],['Sư phạm toán','Sư phạm Toán','Sư phạm Toán học','Sư phạm Toán học ','SP. Toán học','Sư phạm Toán học ']
,['Sư phạm ngữ văn','Sư phạm Văn'],['Sư phạm địa','Sư phạm Địa lý','Địa lý Kinh tế phát triển vùng','Địa lý học'],['Thiết kế thời trang']
,['Quản lý Công nghiệp','quản lý công nghiệp','Quản lý công nghiệp','Quản lý công nghiệp ']
,['Điều khiển và tự động hóa','Điều Khiển tự động hóa','Khoa Học Vật Liệu','đk tự động hóa','Kỹ thuật đk và tđh','Kỹ thuật điều khiển và tự động hóa','Kỹ thuật điều khiển và tự động hoá','Kỹ thuật điều khiển và tự động hóa ','KTĐK&TĐH','Điều khiển và Tự động hóa','Điều Khiển Tự Động','Tự động hóa','Kỹ Thuật Điều Khiển & Tự Động Hóa']
,['Vật Lý kĩ thuật','Vật lý kĩ thuật','Vật lý kỹ thuật','Kỉ thuật']
,['Kinh doanh nông nghiệp','Chăn nuôi','Kinh tế nông nghiệp','Kinh tế Nông nghiệp']
,['Vân tải'],['Triết học'],['Phương đông học','Đông Phương Học','Đông phương học ','Đông phuong'],['Y học cổ truyền','Y học Cổ truyền','y học cổ truyền','Bác sũ Y học cổ truyền','Bác sí Y học cổ truyền','Yhct']
,['Du lịch','Du Lịch','Du Lịch ','Quản trị dịch vụ du lịch và lữ hành','Quản trị Kinh doanh Tổng hợp','Quản trị du lích','Quản trị du lịch','Quan trị dịch vụ du lịch và lữ hành','Hướng dân viên ','Hướng dẫn viên du lịch','Hướng dẫn du lịch','Quản trị khách sạn','Quản Trị Khách Sạn','Quản trị khách sạn ','Quản trị khách sạn và khu du lịch','Quản trị khách sạn và khu du lịch    ','Quản Trị Nhà Hàng Dịch vụ ăn uống','Ngành quản trị nhà hàng - khách sạn','Du lịch và quản lý du lịch ','Du lịch ']
,['Khoa học đất'],['Kinh doanh quốc tế','Kinh Doanh Quốc Tế','Kinh Doanh quốc tế'],['Nuôi trồng thủy sản',' Nuôi trồng thủy sản','Nuôi Trồng Thủy Sản','Nuôi trồng thủy sản ','Bệnh học thủy sản ','Bệnh học thuỷ sản ','Bệnh học thủy sản','Bệnh học Thuỷ sản'],['Thú y', 'Bác sí Thú Y','Thú Y','Bác sĩ Thú Y']
,['Toán ứng dụng'],['Việt Nam học',' Việt Nam học ','Việt Nam Học ','Việt Nam học','Văn hóa học','Văn Hóa Học','Truyển thông văn hoá ','Văn hoá học ','Văn Hoá học'],['Kiểm toán','Kinh toán kiểm toán','Kế toán Kiểm toán','Kế toán tài chính','Kế toán kiểm toán']
,['Marketing','Maketing '],['Giáo dục công dân'],['Công nghệ sinh học','Công nghệ sinh học ','Sinh hóa'],['SInh hóa']
,['Kinh doanh thương mại'],['Ngôn ngữ Pháp'],['Sinh học ứng dụng'],
['Công nghệ chế biến thủy sản','Công nghệ chế biến thủy sản '],['Deutsch lernen','Tiếng Đức'],['Giáo dục thể chất','Giáo dục thể chất ','Giáo Dục Thể Chất'],['Sư phạm hóa học','Sư phạm Hóa học']
,['Thương mại điện tử','Thuơng mại điện tử','Thuơng mại điện tử ','Tmđt','Tmdt']
,['Tiếng Trung Quốc Thương Mại ','Ngôn ngữ Ttrung','Ngôn ngữ Trung'],['Báo chí','Quan hệ công chúng','Quan hệ công chúng ','Quan Hệ Quốc Tế','Quan hệ quốc tế'],['Logistics'],['Ngôn ngữ Nhật','Ngôn ngữ Hàn','Nhật Bản học','Tiếng Nhật','Ngôn ngữ nhật']
,['Sư phạm Tiểu học ','Sư phạm tiểu học','Sư phạm'],['Tâm lý học'],['Kiến trúc','Kiến Trúc','Kiến trúc Nội thất'],['Giáo Dục Mầm Non ',' Giáo dục Mầm non   ',' Giáo dục Mầm non  ','Giáo dục Mầm non',' Giáo dục Mầm non ',' Giáo dục mầm non','Giáo dục mầm non','sư phạm mầm non','Gíao dục nầm non','Mầm non','Giáo dục Mần non','không tiết lộ','GDMN','Gdmn','Giáo dục Mầm Non','Giáo dục Mầm non','Giáo dục mầm non ','Giáo dục mần non','Giáo dục mầm non','Giáo dục Mầm non  ','Gíao dục mầm non','Giáo Dục Mầm Non','Giáo dục mần non ','giáo dục mầm non']
,['Bartender hoặc đầu bếp','Đầu bếp']
]

major_names = ['Khoa học máy tính','Công nghệ thông tin','Truyền thông mạng máy tính',
'Công nghệ phần mềm','Tin học ứng dụng','Hệ thống thông tin','Chính trị học',
'Quản trị kinh doanh','Quản lý đất đai','Kế toán','Luật','Khoa học môi trường',
'Khoa học xã hội và nhân văn','Xây dựng','Điều dưỡng','Dược sĩ','Văn học',
'Ngôn ngữ anh','Y đa khoa','Y học dự phòng','Thông tin học','Răng hàm mặt',
'Mỹ thuật đa phương tiện','Kỹ Thuật điện - điện tử','Dệt may','Sư phạm','Cơ khí',
'Hóa học','Cao đẳng y tế','Quân đội','Bảo vệ thực vật','Công nghệ thực phẩm',
'Nông học','Tài chính ngân hàng','Sư phạm toán','Sư phạm ngữ văn','Sư phạm địa',
'Thiết kế thời trang','Quản lý công nghiệp','Điều Khiển tự động hóa','Vật Lý kỹ thuật',
'Kinh tế nông nghiệp','Vận tải','Triết học','Phương đông học','Y học cổ truyền',
'Quản trị du lịch lữ hành','Khoa học đất','Kinh doanh quốc tế','Nuôi trồng thủy sản',
'Thú y','Toán ứng dụng','Việt Nam học','Kiểm toán','Marketing',
'Giáo dục công dân','Công nghệ sinh học','Sinh hóa','Kinh doanh thương mại','Ngôn ngữ Pháp',
'Sinh học ứng dụng','Công nghệ chế biến thủy sản','Tiếng Đức','Giáo dục thể chất','Sư phạm hóa học'
,'Thương mại điện tử','Tiếng Trung','Báo chí','Logistics','Ngôn ngữ Nhật','Sư phạm tiểu học','Tâm lý học'
,'Kiến Trúc','Giáo dục mầm non','Đầu bếp']

for i in range(len(dataset['Major'])):
    for j in range(len(major_names)):
        if dataset['Major'].values[i] is None:
            dataset['Major'].values[i] = "Unknown"
        elif dataset['Major'].values[i] in arr_major_names[j]:
            dataset['Major'].values[i] = major_names[j]


# le = preprocessing.LabelEncoder()
# le.fit(['Nam','Nữ'])
# le.classes_
# dataset['gender'] = le.transform(dataset['gender'])

list_major_names = dataset.iloc[:,2].drop_duplicates()
list_major_names = list_major_names.dropna()
list_major_names = dataset['Major']


#Xử lý dữ liệu Sở thích

array_interst_users = []


for i in range(dataset.shape[0]):
    txt = dataset.iloc[i,1]
    x = txt.split(',')
    array_interst_users.append(x)

for i in range(len(array_interst_users)):
    for j in range(len(array_interst_users[i])):
        array_interst_users[i][j] = array_interst_users[i][j].lstrip()

# df = pd.DataFrame(np.array(array_interst_users).reshape(1,1062), columns = list("1"))
# array_interst_users = pd.DataFrame(array_interst_users)
# array_interst_users
#Danh sách sở thích trước khi chuẩn hóa
interest_handling = [
['Chơi game','Chơi game','chơi game','Game','Game ']
,['Thể thao']
,['Kiến trúc','Biên tập viên']
,['Chụp ảnh','chụp ảnh']
,['dancing','Nhảy']
,['Âm nhạc']
,['Tin học','Phóng viên','...','phóng viên','phóng viên ']
,['phượt','Moto','Dulịch']
]

interested = dataset['Interest']
#Danh sách sở thích trước sau chuẩn hóa
interest_processed = ['chơi game','Thể thao','Biên tập viên','Chụp ảnh','dancing','Âm nhạc','phóng viên','Du lịch']

for i in range(len(array_interst_users)):
    for j in range(len(array_interst_users[i])):
        if array_interst_users[i][j] is None:
            array_interst_users[i][j] = "Unknown"
        else:
            for k in range(len(interest_handling)):
                if array_interst_users[i][j] in list(interest_handling[k]):
                    array_interst_users[i][j] = interest_processed[k]

#   Danh sách sở thích chung (Tất cả sở thích)
array_interest = array_interst_users[0]# 11 sở thích đó 

for i in range(len(array_interst_users)):
    for j in range(len(array_interst_users[i])):
        if array_interst_users[i][j] not in list(array_interest):
            array_interest.append(array_interst_users[i][j])

array_split = []
for i in range(len(array_interest)):
    tam = []
    for j in range(len(array_interst_users)):
        if array_interest[i] in list(array_interst_users[j]):
            tam.append(1)
        else:
            tam.append(0)
    array_split.append(tam)    
    

for i in range(len(array_interest)):
    dataset[array_interest[i]] = array_split[i]

# dataset = dataset.drop('interest')

#điểm 
del dataset['Interest'] # Sau khi sử lý sở thích xong thì xóa cột sở thích trong tập dữ liệu gốc

score_all = dataset.iloc[:,[4,5,6,7,8,9,10]]
# mc = score_all['math_score'].drop_duplicates()
#điểm trước chuẩn hóa

#Math_score---------------
score_handling_math = [
    ['Dưới 5','Đang học lớp 11','Em chưa học tới ạ']
    ,['Từ 5 - dưới 6.5']
    ,['Từ 6.5 - dưới 8','5.0-7.3','Từ 6.5 -  dưới 8']
    ,['Từ 8 - dưới 9','Từ 8 -  dưới 9']
    ,['Từ 9 - 10']
]
#Physic_score---------------
score_handling_physic = [
    ['Dưới 5','Đang học lớp 11','Em chưa học tới ạ']
    ,['Từ 5 - dưới 6.5']
    ,['Từ 6.5 - dưới 8','5.0-7.9','Từ 6.5 -  dưới 8']
    ,['Từ 8 - dưới 9','Từ 8 -  dưới 9']
    ,['Từ 9 - 10']
]
#chemistry_score---------------
score_handling_chemistry = [
    ['Dưới 5','Đang học lớp 11','Em chưa học tới ạ']
    ,['Từ 5 - dưới 6.5']
    ,['Từ 6.5 - dưới 8','5.0-7.9','Từ 6.5 -  dưới 8']
    ,['Từ 8 - dưới 9','Từ 8 -  dưới 9']
    ,['Từ 9 - 10']
]
#score_eng---------------
score_handling_eng = [
    ['Dưới 5','Đang học lớp 11','Em chưa học tới ạ']
    ,['Từ 5 - dưới 6.5']
    ,['Từ 6.5 - dưới 8','5.0-7.9','Từ 6.5 -  dưới 8']
    ,['Từ 8 - dưới 9','Từ 8 -  dưới 9']
    ,['Từ 9 - 10']
]
#literature_score--------------
score_handling_litrature = [
    ['Dưới 5','Đang học lớp 11','Em chưa học tới ạ']
    ,['Từ 5 - dưới 6.5']
    ,['Từ 6.5 - dưới 8','5.0-7.9','Từ 6.5 -  dưới 8']
    ,['Từ 8 - dưới 9','Từ 8 -  dưới 9']
    ,['Từ 9 - 10']
] 
#history_score--------------
score_handling_history = [
    ['Dưới 5','Đang học lớp 11','Em chưa học tới ạ']
    ,['Từ 5 - dưới 6.5']
    ,['Từ 6.5 - dưới 8','5.0-7.9','Từ 6.5 -  dưới 8']
    ,['Từ 8 - dưới 9','Từ 8 -  dưới 9']
    ,['Từ 9 - 10']
]
#geography_score--------------
score_handling_geography = [
    ['Dưới 5','Đang học lớp 11','Em chưa học tới ạ']
    ,['Từ 5 - dưới 6.5']
    ,['Từ 6.5 - dưới 8','5.0-7.9','Từ 6.5 -  dưới 8']
    ,['Từ 8 - dưới 9','Từ 8 -  dưới 9']
    ,['Từ 9 - 10']
]


score_processed = [2.5,5.75,7.25,8.5,9.5]


def transcore(dataset, list_values, list_values_tranform, feature_name ): 
    for i in range(len(dataset)):
        for j in range(len(list_values)):
            if dataset[feature_name][i] in list(list_values[j]):
                dataset[feature_name][i] = list_values_tranform[j]
    dataset[feature_name].fillna(2.5)
    return dataset

dataset = transcore(dataset, score_handling_math, score_processed, 'Maths_score')
dataset = transcore(dataset, score_handling_physic, score_processed, 'Physics_score')
dataset = transcore(dataset, score_handling_chemistry, score_processed, 'Chemistry_score')
dataset = transcore(dataset, score_handling_eng, score_processed, 'English_score')
dataset = transcore(dataset, score_handling_litrature, score_processed, 'Literature_score')
dataset = transcore(dataset, score_handling_history, score_processed, 'History_score')
dataset = transcore(dataset, score_handling_geography, score_processed, 'Geography_score')
 
#Chuẩn hóa tới lý do chọn ngành
reason_handling = [['Chọn đại','Chọn đại : )','Đăng ký đại','Dòng đời xô đẩy ','Lỡ tay chọn']
,['Sở thích, năng lực','Tất cả','Yêu thích công nghệ và được tư vấn từ người thân trong gia đình.','Sở thích, năng lực, thu nhập, nhu cầu thị trường','Học để hiểu biết']]

reason_processed = ['Năng lực bản thân','Sở thích']

dataset = transcore(dataset, reason_handling, reason_processed, 'RFCM')
dataset = dataset.reset_index(drop = True)

#chuẩn hóa dữ liệu lý do chọn ngành thành số
#
dataset['RFCM'] = dataset['RFCM'].fillna('Sở thích')
dataset['Time'] = dataset['Time'].fillna(3)
dataset['Maths_score'] = dataset['Maths_score'].fillna(5)
dataset['Physics_score'] = dataset['Physics_score'].fillna(5)
dataset['Chemistry_score'] = dataset['Chemistry_score'].fillna(5)
dataset['English_score'] = dataset['English_score'].fillna(5)
dataset['Literature_score'] = dataset['Literature_score'].fillna(5)
dataset['History_score'] = dataset['History_score'].fillna(5)
dataset['Geography_score'] = dataset['Geography_score'].fillna(5)

dataset = dataset.dropna()


dataset = transform_encoder('RFCM',['Sở thích', 'Năng lực bản thân', 'Nguyện vọng của gia đình', 'Tư vấn từ thầy cô, bạn bè', 'Nhu cầu cao của thị trường đối với ngành học (Khả năng tìm việc làm cao)', 'Mức thu nhập của gia đình'])

dataset = dataset[['Gender', 'RFCM', 'Time','Influence','Du lịch','Thể thao điện tử','Phim ảnh','Âm nhạc'
,'Thời trang','Đọc sánh','Hội họa - mỹ thuật', 'Thể thao', 'Chụp ảnh','chơi game','Biên tập viên','phóng viên'
,'Viết','dancing','Uống trà sữa cùng hội chị em.','Maths_score','Physics_score','Chemistry_score'
,'English_score','Literature_score','History_score','Geography_score','Major']]

# export_csv = dataset.to_csv(
#     r"D:\LUANVAN\dataset.csv",index = None, header = True
# )


#------------------------------

data = pd.read_csv("./dataset.csv")
# data_project = data.drop('tk')
data_project = data
data_project = data_project.sample(frac=1)

#   Chia dữ liệu train 
data_train = data_project.iloc[388:,:]
data_train_dm = data_train.iloc[:,:4]
data_train_ctb = data_train.iloc[:,4:]

#   Chia dữ liệu test 
data_test = data_project.iloc[:388,:]
data_test_dm = data_test.iloc[:,:4]
# data_train_dm1 = data_test.iloc[:,[1,3]]
data_test_ctb = data_test.iloc[:,4:]

#KMeans Cluster
clusters = 12 #tại test 5, 7, 9, 12, 15 thì thấy 12 ok
from sklearn.cluster import KMeans
model = KMeans(n_clusters=clusters, random_state=0).fit(data_train_dm) 
#-----------------------------
# y_means = model.fit_predict(data_train_dm1) 
import matplotlib.pyplot as plt
import random