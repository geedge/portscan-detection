import itertools
import time
from sklearn.preprocessing import normalize
from sklearn.metrics import accuracy_score, classification_report
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import sklearn.metrics as sm
from scipy.spatial.distance import cdist

label = ['BENIGN', 'Bot', 'DDos', 'GlodenEye', 'Dos Hulk',
         'Slowhttp', 'SSH', 'FTP', 'PortScan', 'slowloris', 'BruteForce', 'XSS']


def HandleData(path):
    list_dir = os.listdir(path)
    fd_data = []
    for it in list_dir:
        data = pd.read_csv(path + '/' + it)
        fd_data.append(data)
    data = pd.concat([fd_data[0], fd_data[1]])
    for it in range(2, len(fd_data)):
        data = pd.concat([data, fd_data[it]])
    data = data.dropna(axis=0, how='any')
    data = data.replace(',,', np.nan, inplace=False)
    data.replace("Infinity", 0, inplace=True)

    data.replace('Infinity', 0.0, inplace=True)
    data.replace('NaN', 0.0, inplace=True)
    data = data.replace([np.inf, -np.inf], np.nan)
    data = data.dropna(axis=0, how='any')
    n_row, n_col = data.shape
    print('row:', n_row, 'col:', n_col)

    return data


def plot_confusion_matrix(cm, classes, normalize=True, title='Confusion matrix', cmap=plt.cm.Blues):
    """
    - cm : 计算出的混淆矩阵的值
    - classes : 混淆矩阵中每一行每一列对应的列
    - normalize : True:显示百分比, False:显示个数
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("显示百分比：")
        np.set_printoptions(formatter={'float': '{: 0.2f}'.format})
        print(cm)
    else:
        print('显示具体数字：')
        print(cm)
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=90)
    plt.yticks(tick_marks, classes)

    plt.ylim(len(classes) - 0.5, -0.5)
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()


def Train(data, decomponent=False):
    #处理空值，无限制
    '''
    data = data.replace([np.inf, -np.inf], np.nan)
    data = data.dropna(axis=0, how='any')
    '''
    
    print(data[' Label'].value_counts())
    #select features
    basic_feature = [' Label','Flow ID',' Source IP',' Source Port',' Destination IP',' Destination Port',' Protocol',' Timestamp']
    tcp_ip_feature = ['FIN Flag Count', ' SYN Flag Count',
                ' RST Flag Count', ' PSH Flag Count', ' ACK Flag Count',
                ' URG Flag Count', ' CWE Flag Count', ' ECE Flag Count','Init_Win_bytes_forward', ' Init_Win_bytes_backward']
    statistical_feature = [' Flow Duration',  'Total Length of Fwd Packets', ' Total Length of Bwd Packets',    
                            ' Fwd Packet Length Max', ' Fwd Packet Length Min', ' Fwd Packet Length Mean', ' Fwd Packet Length Std',
                            'Bwd Packet Length Max', ' Bwd Packet Length Min',' Bwd Packet Length Mean', ' Bwd Packet Length Std', 
                            'Fwd IAT Total',' Fwd IAT Mean', ' Fwd IAT Std',' Fwd IAT Max', ' Fwd IAT Min', 
                            'Bwd IAT Total', ' Bwd IAT Mean', ' Bwd IAT Std', ' Bwd IAT Max', ' Bwd IAT Min',
                            ' Packet Length Mean', ' Packet Length Std', ' Packet Length Variance', ' Min Packet Length', ' Max Packet Length',
                            'Fwd Packets/s',' Bwd Packets/s',
                            ' Flow Packets/s' ,'Flow Bytes/s']
    middle_feature = ['pre_label','port_flow']
    add_feature = ['cu_ipnum','cu_dip','cu_dport','cu_target']
    x_columns = tcp_ip_feature + statistical_feature + add_feature  
    print(x_columns)
    x = data[x_columns].values
    x = normalize(x, axis=0, norm='max')
    dummies = pd.get_dummies(data[' Label'])
    #dummies = pd.get_dummies(data['label'])
    outcomes = dummies.columns
    print(outcomes)
    num_classes = len(outcomes)
    print('[traffic] 类别数:', num_classes)
    y = dummies.values
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=20)

    return x_train, y_train, x_test, y_test

def SVM(train_X, train_Y, test_X, test_Y):
    print('[SVM] train ...')
    train_Y = [np.where(r == 1)[0][0] for r in train_Y]
    test_Y = [np.where(r == 1)[0][0] for r in test_Y]
    t1 = time.time()
    clf = svm.SVC(decision_function_shape='ovr', max_iter=1000, kernel='rbf')
    model = clf.fit(train_X, train_Y)
    y_hat = model.predict(test_X)
    acc = accuracy_score(test_Y, y_hat)
    t2 = time.time()
    print('acc:', acc)
    print('using time:', t2 - t1, 'sec')
    matrix = sm.confusion_matrix(test_Y, y_hat)
    print(matrix)
    report = classification_report(test_Y, y_hat)
    print(report)
    print('-' * 20)
    '''
    import pickle
    with open('svm.pickle','wb') as f:
        pickle.dump(clf,f)
    '''
    #plot_confusion_matrix(matrix, label, True, 'SVM Confusion matrix')

def Elbow_kmeans(X):
    K = range(1, 40)
    meandistortions = []
    for k in K:
        kmeans = KMeans(n_clusters=k)
        kmeans.fit(X)
        meandistortions.append(sum(np.min(cdist(X, kmeans.cluster_centers_, 'euclidean'), axis=1))/X.shape[0])
    plt.plot(K, meandistortions, 'bx-')
    plt.xlabel('k')
    plt.ylabel('Average Dispersion')
    plt.title('Selecting k with the Elbow Method')
    plt.show()

def Kmeans(data):        
    print(data[' Label'].value_counts())
    basic_feature = [' Label','Flow ID',' Source IP',' Source Port',' Destination IP',' Destination Port',' Protocol',' Timestamp']
    tcp_ip_feature = ['FIN Flag Count', ' SYN Flag Count',
                ' RST Flag Count', ' PSH Flag Count', ' ACK Flag Count',
                ' URG Flag Count', ' CWE Flag Count', ' ECE Flag Count','Init_Win_bytes_forward', ' Init_Win_bytes_backward']
    statistical_feature = [' Flow Duration',  'Total Length of Fwd Packets', ' Total Length of Bwd Packets',
                            ' Fwd Packet Length Max', ' Fwd Packet Length Min', ' Fwd Packet Length Mean', ' Fwd Packet Length Std',
                            'Bwd Packet Length Max', ' Bwd Packet Length Min',' Bwd Packet Length Mean', ' Bwd Packet Length Std',
                            ' Fwd IAT Mean', ' Fwd IAT Std',' Fwd IAT Max', ' Fwd IAT Min',
                            'Bwd IAT Total', ' Bwd IAT Mean', ' Bwd IAT Std', ' Bwd IAT Max', ' Bwd IAT Min',
                            ' Packet Length Mean', ' Packet Length Std', ' Packet Length Variance', ' Min Packet Length', ' Max Packet Length',
                            'Fwd Packets/s',' Bwd Packets/s',
                            ' Flow Packets/s' ,'Flow Bytes/s']
    x_columns = tcp_ip_feature + statistical_feature 
    x = data[x_columns].values
    x = normalize(x, axis=0, norm='max')
    #处理标签
    dummies = pd.get_dummies(data[' Label'])
    #dummies = pd.get_dummies(data['label'])
    outcomes = dummies.columns
    print(outcomes)
    num_classes = len(outcomes)
    print('[traffic] 类别数:', num_classes)
    y = dummies.values
    Y = y[:,0]
    #聚类
    Z = pso_kmeans(x,Y)
    return Z

def pso_kmeans(X,Y):
    from pso import ParticleSwarmOptimizedClustering
    from particle import quantization_error, calc_sse
    from utils import normalize
    from kmeans import KMeans
    from sklearn.metrics import silhouette_score
    from scipy.spatial.distance import cdist
    distortions = []
    K = [30]
    for k in K: 
        pso_rep = ParticleSwarmOptimizedClustering(
        n_cluster=k, n_particles=10, data=X, hybrid=True, max_iter=50, print_debug=2000)
        pso_rep.run()
        pso_kmeans = KMeans(n_cluster=k, init_pp=False, seed=2022)
        pso_kmeans.centroid = pso_rep.gbest_centroids.copy()
        predicted_pso_rep = pso_kmeans.predict(X)
    
        silhouette = silhouette_score(X, predicted_pso_rep)
        sse = calc_sse(centroids=pso_rep.gbest_centroids, data=X, labels=predicted_pso_rep)
        quantization = pso_rep.gbest_score
        Z = pso_kmeans.predict(X)
        #
        distortions.append(sum(np.min(cdist(X, pso_kmeans.centroid, 'euclidean'), axis=1)) / X.shape[0])
        print(pso_kmeans.centroid)
        print(type(Z))
        kmeansR = pd.crosstab(Y,Z)
        maxVal = kmeansR.idxmax()
        print(kmeansR,"\n\n")
    #保存
    import pickle
    with open('train_model_sta.pkl', 'wb') as f:
        pickle.dump(pso_kmeans, f)
    '''
    #加载
    import pickle
    pso_kmeans =  pickle.load(file=open('train_model_all.pkl', 'rb'))
    '''
    Z = pso_kmeans.predict(X)
    kmeansR = pd.crosstab(Y,Z)
    maxVal = kmeansR.idxmax()
    print(kmeansR,"\n\n")
    return Z

result_dict = {}
def formula(x):
    if x in result_dict:
        return result_dict[x]
    else:
        return 0

def process_data(data,pre_label):
    data['pre_label'] = pre_label
    x_columns = ['pre_label',' Destination Port']
    label_num = data['pre_label'].unique()
    for i in range(len(label_num)):
        tmp_data = data.loc[data['pre_label'] == i]
        flow_num = len(tmp_data)
        port_num = len(tmp_data[' Destination Port'].unique())
        tmp_result = port_num
        result_dict[i] = tmp_result
    print(result_dict)
    data['port_flow'] = ''
    data['port_flow'] = data.apply(lambda row: formula(row['pre_label']), axis=1)
    return data

def add_feature(data):
    '''
    增加特征
    '''    
    #该扫描结点（IP)所在簇连接的taget的数量
    cu_target = {}
    for index,row in data.iterrows():
        if row['pre_label'] not in cu_target:
            cu_target[row['pre_label']] = set()            
        cu_target[row['pre_label']].add(str(row[' Destination IP'])+'-'+str(row[' Destination Port']))
    data['cu_target'] = ''
    def formula_cu_target(x):
        if x in cu_target:
            return len(cu_target[x])
        else:
            return 0
    data['cu_target'] = data.apply(lambda row: formula_cu_target(row['pre_label']), axis=1)
    #该扫描结点所在簇的结点（IP)数量
    cu_ipnum = {}
    for index,row in data.iterrows():
        if row['pre_label'] not in cu_ipnum:
            cu_ipnum[row['pre_label']] = set()            
        cu_ipnum[row['pre_label']].add(row[' Source IP'])
    data['cu_ipnum'] = ''
    def formula_cu_ipnum(x):
        if x in cu_ipnum:
            return len(cu_ipnum[x])
        else:
            return 0
    data['cu_ipnum'] = data.apply(lambda row: formula_cu_ipnum(row['pre_label']), axis=1)
    
    #该扫描结点（IP)所在簇连接的目标IP的数量
    cu_dip = {}
    for index,row in data.iterrows():
        if row['pre_label'] not in cu_dip:
            cu_dip[row['pre_label']] = set()            
        cu_dip[row['pre_label']].add(str(row[' Destination IP']))
    data['cu_dip'] = ''
    def formula_cu_dip(x):
        if x in cu_dip:
            return len(cu_dip[x])
        else:
            return 0
    data['cu_dip'] = data.apply(lambda row: formula_cu_dip(row['pre_label']), axis=1)
    #该扫描结点（IP)所在簇连接的目标端口的数量
    cu_dport = {}
    for index,row in data.iterrows():
        if row['pre_label'] not in cu_dport:
            cu_dport[row['pre_label']] = set()            
        cu_dport[row['pre_label']].add(str(row[' Destination Port']))
    data['cu_dport'] = ''
    def formula_cu_dport(x):
        if x in cu_dport:
            return len(cu_dport[x])
        else:
            return 0
    data['cu_dport'] = data.apply(lambda row: formula_cu_dport(row['pre_label']), axis=1)

    return data

def main():
    data = HandleData('H:\dataset\CICIDS2017\pcap\CIC\cic_trafficlabeling\input\Test')
    pre_label = Kmeans(data)
    result_data = process_data(data,pre_label)
    result_data = add_feature(result_data)
    train_X, train_Y, test_X, test_Y= Train(result_data)
    SVM(train_X, train_Y, test_X, test_Y)
    # MLP(train_X, train_Y, test_X, test_Y)

if __name__ == '__main__':
    main()
