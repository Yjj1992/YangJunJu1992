import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#思考一下，是否可直接与已知中心点的距离之和的最大值来选下一个中心点？
def centroids_initial(df, k):
    centroids = {
            i:[0,0]
            for i in range(k)
            }
    for i in centroids.keys():
        if i == list(centroids.keys())[0]:
            ei=np.random.randint(0,df['x'].size)
            centroids[i][0] = df.loc[ei]['x']
            centroids[i][1] = df.loc[ei]['y']
        else:
            df['distance_from_last_centroids']=(df['x'] - centroids[i-1][0])**2+(df['y'] - centroids[i-1][1])**2
            sum_temp=np.sum(df['distance_from_last_centroids'])
            df['distance_from_last_centroids']=df['distance_from_last_centroids']/sum_temp
            temp1=np.zeros(df.shape[0])
            for j in range(df.shape[0]):
                temp1[j]=df.loc[j]['distance_from_last_centroids']
                if j!=0:
                    temp1[j]+=temp1[j-1]
            #注意datafram要对某个定位值进行迭代会发生错误，只能对某一列进行迭代，或者新建列的同时对列中的某个值进行赋值
            df['distance_from_last_centroids']=temp1       
            threshold_temp=np.random.random()
            idxmin=df[df['distance_from_last_centroids']>=threshold_temp].idxmin(axis=0)['distance_from_last_centroids']
            centroids[i][0] = df.loc[idxmin]['x']
            centroids[i][1] = df.loc[idxmin]['y']
    return centroids

def assignment(df, centroids, colmap):
    for i in centroids.keys():
        # sqrt((x1 - x2)^2 - (y1 - y2)^2)
        df['distance_from_{}'.format(i)] = (
            np.sqrt(
                (df['x'] - centroids[i][0]) ** 2
                + (df['y'] - centroids[i][1]) ** 2
            )
        )
    distance_from_centroid_id = ['distance_from_{}'.format(i) for i in centroids.keys()]
    #下面的idxmin返回最小值的索引，如果多个最小值，则返回最先遇到的，参数中axis=1表示跨列比较，就是在同一行中进行比较
    #注意 df 的列标签是‘字符串’，因此下方返回的 idx 也是字符串
    df['closest'] = df.loc[:, distance_from_centroid_id].idxmin(axis=1)
    #map函数的第一个参数是执行程序，第二个参数是需要执行的序列
    #lambda x : func 的写法是匿名函数的意思，x 指代 func 中的变量，多个变量见用逗号隔开，func 就是要执行的函数
    df['closest'] = df['closest'].map(lambda x: int(x.lstrip('distance_from_')))
    df['color'] = df['closest'].map(lambda x: colmap[x])
    return df

def update(df, centroids):
    for i in centroids.keys():
        centroids[i][0] = np.mean(df[df['closest'] == i]['x'])
        centroids[i][1] = np.mean(df[df['closest'] == i]['y'])
    return centroids

def main():
    # step 0.0: generate source data
    df = pd.DataFrame({
        'x': [12, 20, 28, 18, 10, 29, 33, 24, 45, 45, 52, 51, 52, 55, 53, 55, 61, 64, 69, 72, 23],
        'y': [39, 36, 30, 52, 54, 20, 46, 55, 59, 63, 70, 66, 63, 58, 23, 14, 8, 19, 7, 24, 77]
    })
    # dataframe 返回一个二维矩阵，
    # 用.loc直接定位
    #
    # 例：
    # data = pd.DataFrame({'A':[1,2,3],'B':[4,5,6],'C':[7,8,9]})
    #
    #     A  B  C
    #  0  1  4  7
    #  1  2  5  8
    #  2  3  6  9
    #
    # 可以用index=["a","b","c"]设置index
    # data = pd.DataFrame({'A':[1,2,3],'B':[4,5,6],'C':[7,8,9]},index=['a','b','c'])
    #
    #     A  B  C
    #  a  1  4  7
    #  b  2  5  8
    #  c  3  6  9


    # step 0.1: generate center
    #np.random.seed(200)    # in order to fix the random centroids
    k = 3
    # centroids[i] = [x, y]
    #centroids = {
    #    i: [np.random.randint(0, 80), np.random.randint(0, 80)]
    #    for i in range(k)
    #}
    centroids = centroids_initial(df, k)
    # step 0.2: assign centroid for each source data
    # for color and mode: https://blog.csdn.net/m0_38103546/article/details/79801487
    # colmap = {0: 'r', 1: 'g', 2: 'b', 3: 'm', 4: 'c'}
    colmap = {0: 'r', 1: 'g', 2: 'b'}
    df = assignment(df, centroids, colmap)

    plt.scatter(df['x'], df['y'], color=df['color'], alpha=0.5, edgecolor='k')
    for i in centroids.keys():
        plt.scatter(*centroids[i], color=colmap[i], linewidths=6)
    plt.xlim(0, 80)
    plt.ylim(0, 80)
    plt.show()

    for i in range(10):

        closest_centroids = df['closest'].copy(deep=True)
        centroids = update(df, centroids)

        plt.scatter(df['x'], df['y'], color=df['color'], alpha=0.5, edgecolor='k')
        for i in centroids.keys():
            plt.scatter(*centroids[i], color=colmap[i], linewidths=6)
        plt.xlim(0, 80)
        plt.ylim(0, 80)
        plt.show()

        df = assignment(df, centroids, colmap)

        if closest_centroids.equals(df['closest']):
            break


if __name__ == '__main__':
    main()
