import pandas as pd 
import numpy as np
from copy import deepcopy

#MinMax正規化
def MinMax(data):
    mi=np.min(data,0)
    ma=np.max(data,0)
    new_data=(data-mi)/(ma-mi+1e-7)
    return new_data

#stockデータの前処理
def preprocess(mode="stock",seq_len=5):
    """
    現段階では、MinMax正規化のみ行っている。
    out:
        data:[batch,seq,dim]
        time:[seq] シーケンス長を補完
        max_seq:scaler 最大シーケンス長
    """
    if mode=="stock":
        data=pd.read_csv("Dataset/stock_data.csv",)
    else:
        exit("error")
    #データの順番を逆にする。
    data=data[::-1]
    #データの正規化(MinMax正規化)
    data=MinMax(data)
    #先頭から連続でseq_length個づつデータをとる。この時、[0,1,2,3,4,..,seq_length-1],[1,2,3,4,..,seq_length],...のようにとってくる。
    new_data=[]
    new_time=[]
    max_seq=0
    for i in range(len(data)-seq_len):
        tmp=data[i:i+seq_len].values
        new_data.append(data[i:i+seq_len].values)
        new_time.append(len(tmp))
        max_seq=max(max_seq,len(tmp))
    #データのシャッフル
    idx=np.random.permutation(len(new_data))
    data = []
    time=[]
    for i in range(len(new_data)):
        data.append(new_data[idx[i]])
        time.append(new_time[idx[i]])
    return data,time,max_seq


def sine_data(num,seq_len,dim):
    """
    data: [num,seq_len,dim]
    time: [num] 各バッチのシーケンスの長さ
    maq_len: 最大シーケンス長
    """
    data=[]
    time=[]
    max_seq=0
    
    for i in range(num):
        #バッチごとの処理
        tmp=[]
        for k in range(dim):
            #各dimでの処理、シーケンスの軸に関してはリスト内で構築可能
            #周波数
            freq=np.random.uniform(0,0.1)
            #位相
            phase=np.random.uniform(0,0.1)
            tmp_data=[np.sin(freq*j+phase) for j in range(seq_len)]
            tmp.append(tmp_data)
        tmp=np.array(tmp)
        #このままでは[num,dim,seq_len]となるので転置
        tmp=tmp.transpose(1,0)
        tmp = (tmp + 1)*0.5
        time.append(len(tmp))
        max_seq=max(max_seq,len(tmp))
        data.append(tmp)
    return data,time,max_seq

def sine_data_alpha(num,seq_len,dim,pad_value):
    """
    data: [num,seq_len,dim]
    time: [num] 各バッチのシーケンスの長さ
    maq_len: 最大シーケンス長

    今回はシーケンス長がランダムになるように設計。
    シーケンス長はseq_len-1,seq_len,seq_len+1のうちランダムに選択される。
    また、今後のためにpad_valueで埋め合わせを行なっている。
    """
    data=[]
    time=[]
    max_seq=0
    for i in range(num):
        tmp=[]
        seq_len_=np.random.randint(seq_len-1, seq_len+2)
        for k in range(dim):
            freq=np.random.uniform(0,0.1)
            phase=np.random.uniform(0,0.1)

            tmp_data=[np.sin(freq*j+phase) for j in range(seq_len_)]
            tmp.append(tmp_data)
        tmp=np.array(tmp)
        tmp=tmp.transpose(1,0)
        tmp = (tmp + 1)*0.5
        time.append(len(tmp))
        max_seq=max(max_seq,len(tmp))
        data.append(tmp)
    
    new_data=[]
    for i in range(num):
        tmp=deepcopy(data[i])
        if len(data[i])<max_seq:
            p=max_seq-len(data[i])
            for j in range(p):
                tmp=np.concatenate([tmp,np.array([[pad_value]*dim])])
        new_data.append(tmp)
    return new_data,time,max_seq

def get_data(args):
    if args.dataset_type == "stock":
        data,time,max_seq=preprocess(mode="stock",seq_len=args.seq_len)
        args.dim=data[0].shape[-1]
    elif args.dataset_type == "normal_sine":
        data,time,max_seq=sine_data(num=args.num_data,seq_len=args.seq_len,dim=args.sine_dim)
        args.dim=args.sine_dim
    elif args.dataset_type =="random_sine":
        data,time,max_seq=sine_data_alpha(num=args.num_data,seq_len=args.seq_len,dim=args.sine_dim,pad_value=args.pad_value)
        args.dim=args.sine_dim
    return data,time,max_seq