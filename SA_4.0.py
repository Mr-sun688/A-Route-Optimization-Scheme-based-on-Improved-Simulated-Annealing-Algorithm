from random import *
import numpy as np
from math import *
from matplotlib import pyplot as plt


def loadDatadet(infile, k):
    f = open(infile, 'r')
    sourceInLine = f.readlines()
    dataset = []
    for line in sourceInLine:
        temp1 = line.strip('\n')
        temp2 = temp1.split(' ')
        dataset.append(temp2)
    for i in range(0, len(dataset)):
        for j in range(k - 1):
            dataset[i].append(float(dataset[i][j + 1]))
        del (dataset[i][0:k])
    return dataset, len(dataset)


# k 列数

def getCityData():
    global N
    infile = 'E:\A大学课程资料\算法基础\TSP数据集\\kroA100.txt'
    citys, N = loadDatadet(infile, 3)
    print(citys)
    citys = np.array(citys)
    print(citys)
    # 计算距离矩阵
    distance = np.zeros((N, N))
    for i in range(N):
        for j in range(0, i + 1):
            t = sqrt((citys[i][0] - citys[j][0]) ** 2 + (citys[i][1] - citys[j][1]) ** 2)
            distance[i][j], distance[j][i] = t, t
    return citys, distance


# 随机生成一个初始解
def init(distance):
    min_distance = 0
    path = []
    for i in range(N):
        path.append(i)
    np.random.shuffle(path)
    path = np.array(path)
    for j in range(len(path) - 1):
        min_distance = min_distance + distance[path[j]][path[j + 1]]
    min_distance = min_distance + distance[path[-1]][path[0]]  # 再加上起点到终点的距离
    best_path = path.copy()
    tmp_distance = min_distance
    tmp_path = best_path.copy()
    return tmp_path, tmp_distance


def swap(path, a, b):
    path[a] = path[a] ^ path[b]
    path[b] = path[a] ^ path[b]
    path[a] = path[a] ^ path[b]


# 产生新解
def updataPath(tmp_path):
    x1 = [0 for q in range(N)]
    # randx,randy = randint(0,N-1),randint(0,N-1)
    # print(tmp_path[randx],tmp_path[randy])
    # while(randx == randy):
    #     randx, randy = randint(0, N - 1), randint(0, N - 1)
    # tmp_path[randx],tmp_path[randy] = tmp_path[randy],tmp_path[randx]
    # print(tmp_path[randx], tmp_path[randy])
    # x1 = tmp_path.copy()
    # print(randx,randy)
    if (np.random.rand() > 0.5):
        n1, n2, n3 = randint(0, N - 1), randint(0, N - 1), randint(0, N - 1)
        n = [n1, n2, n3]
        n.sort()
        n1, n2, n3 = n
        x1[0:n1] = tmp_path[0:n1]
        x1[n1:n3 - n2 + n1] = tmp_path[n2 + 1:n3 + 1]
        x1[n3 - n2 + n1:n3 + 1] = tmp_path[n1:n2 + 1]
        x1[n3 + 1:N] = tmp_path[n3 + 1:N]
        # n1, n2 = randint(0, N - 1), randint(0, N - 1)
        # while(n1 == n2):
        #     n1, n2 = randint(0, N - 1), randint(0, N - 1)
        # tmp_path[n1],tmp_path[n2] =tmp_path[n2],tmp_path[n1]
        # x1 = tmp_path.copy()

    else:
        # 将某一段序列置反
        n1, n2 = randint(0, N - 1), randint(0, N - 1)
        n = [n1, n2]
        n.sort()
        n1, n2 = n
        # n1为0单独写
        if n1 > 0:
            x1[0:n1] = tmp_path[0:n1]
            x1[n1:n2 + 1] = tmp_path[n2:n1 - 1:-1]
            x1[n2 + 1:N] = tmp_path[n2 + 1:N]
        else:
            x1[0:n1] = tmp_path[0:n1]
            x1[n1:n2 + 1] = tmp_path[n2::-1]
            x1[n2 + 1:N] = tmp_path[n2 + 1:N]

    # 计算新解距离
    s = 0
    for j in range(len(x1) - 1):
        s = s + distance[x1[j]][x1[j + 1]]
    s = s + distance[x1[-1]][x1[0]]
    return x1, s


# 改进产生新解 p 为结束扰动的概率
def updataPath2(tmp_path, tmp_distance, T, value, min):
    x1 = [0 for q in range(N)]
    goodMax = int(0.0015 * T)  # 择优次数
    good = 0
    while (1):
        if (np.random.rand() > 0.5):
            n1, n2, n3 = randint(0, N - 1), randint(0, N - 1), randint(0, N - 1)
            n = [n1, n2, n3]
            n.sort()
            n1, n2, n3 = n
            x1[0:n1] = tmp_path[0:n1]
            x1[n1:n3 - n2 + n1] = tmp_path[n2 + 1:n3 + 1]
            x1[n3 - n2 + n1:n3 + 1] = tmp_path[n1:n2 + 1]
            x1[n3 + 1:N] = tmp_path[n3 + 1:N]
            # 计算新解距离
            s = 0
            for j in range(len(x1) - 1):
                s = s + distance[x1[j]][x1[j + 1]]
            s = s + distance[x1[-1]][x1[0]]
            if (s <= tmp_distance):
                good = good + 1
                tmp_distance = s
        else:
            # 将某一段序列置反
            n1, n2 = randint(0, N - 1), randint(0, N - 1)
            n = [n1, n2]
            n.sort()
            n1, n2 = n
            # n1为0单独写
            if n1 > 0:
                x1[0:n1] = tmp_path[0:n1]
                x1[n1:n2 + 1] = tmp_path[n2:n1 - 1:-1]
                x1[n2 + 1:N] = tmp_path[n2 + 1:N]
            else:
                x1[0:n1] = tmp_path[0:n1]
                x1[n1:n2 + 1] = tmp_path[n2::-1]
                x1[n2 + 1:N] = tmp_path[n2 + 1:N]
            s = 0
            for j in range(len(x1) - 1):
                s = s + distance[x1[j]][x1[j + 1]]
            s = s + distance[x1[-1]][x1[0]]
            if (s <= tmp_distance):
                good = good + 1
                tmp_distance = s
        if (tmp_distance < min):
            value.append(tmp_distance)
        else:
            value.append(min)
        if (good >= goodMax): break
    return x1, s


def sa_small_change(tmp_path, tmp_distance, value):
    global N
    min_distance = tmp_distance  # 最佳距离
    best_path = tmp_path
    for i in range(5):
        for k in range(N):
            swap(tmp_path, k, (k + 1) % N)
            # 计算新解距离
            s = 0
            for j in range(len(tmp_path) - 1):
                s = s + distance[tmp_path[j]][tmp_path[j + 1]]
            s = s + distance[tmp_path[-1]][tmp_path[0]]
            if s < min_distance:
                min_distance = s
                best_path = tmp_path.copy()
            value.append(min_distance)
    return best_path, min_distance


def sa_again(tmp_path, tmp_distance, value):
    # 初始化参数
    iteration1 = 10000  # 外循环迭代次数
    T0 = 1000  # 初始温度，取大些
    Tf = 1  # 截止温度，可以不用
    alpha = 0.99  # 温度更新因子
    iteration2 = 20  # 内循环迭代次数
    min_distance = tmp_distance  # 最佳距离
    best_path = tmp_path
    now_value = []  # 当次最优解

    num_itera = 0  # 记录迭代次数
    M = 0  # 记录进入if的次数
    N = 188
    plt.ion()
    while (T0 > Tf):
        p = 0
        for k in range(iteration2):
            M = M + 1

            # 生成新解
            x1, s = updataPath(tmp_path)
            # x1, s = updataPath2(tmp_path, tmp_distance, T0,value,min_distance)
            # [12, 13, 15, 14, 21, 20, 16, 19, 202, 18, 17, 10, 11, 35, 36, 37, 31, 30, 218, 215, 75, 73, 74, 97, 98, 99, 101, 70, 72, 71, 69, 67, 68, 103, 219, 102, 100, 90, 91, 89, 86, 87, 88, 104, 105, 106, 107, 108, 109, 111, 63, 64, 66, 65, 62, 61, 60, 57, 56, 50, 48, 49, 51, 54, 55, 53, 52, 40, 39, 38, 9, 8, 7, 6, 5, 4, 3, 2, 0, 199, 197, 196, 194, 42, 41, 43, 45, 193, 217, 44, 47, 192, 195, 191, 190, 223, 198, 132, 204, 188, 26, 46, 224, 189, 206, 1, 58, 59, 115, 116, 187, 186, 117, 222, 114, 113, 112, 110, 120, 118, 185, 184, 183, 119, 174, 121, 122, 123, 124, 167, 168, 169, 170, 171, 172, 181, 180, 173, 179, 178, 175, 177, 176, 144, 143, 142, 200, 141, 140, 139, 138, 137, 135, 182, 134, 133, 214, 163, 212, 157, 162, 136, 160, 161, 159, 158, 156, 155, 154, 153, 152, 145, 146, 147, 148, 151, 150, 149, 211, 213, 166, 125, 126, 165, 164, 127, 128, 221, 131, 130, 210, 129, 209, 85, 84, 83, 82, 81, 220, 93, 92, 95, 208, 94, 80, 79, 96, 77, 78, 76, 216, 205, 201, 29, 34, 32, 33, 28, 27, 203, 25, 24, 207, 23, 22]
            # 4293.333761570366
            # [12, 13, 15, 14, 21, 20
            now_value.append(s)
            # 判断是否更新解
            if s <= tmp_distance:  # 如果小于 直接接受
                M = 0
                tmp_distance = s
                tmp_path = x1.copy()
                num_itera += 1
                p = p + 1
            if s > tmp_distance:  # 如果大于 以一定概率接受
                deltaf = s - tmp_distance
                if random() < exp(-deltaf / T0):
                    M = 0
                    tmp_distance = s
                    tmp_path = x1.copy()
                    num_itera += 1
            if s < min_distance:  # 记录最优解
                min_distance = s
                best_path = x1.copy()
                # 动画演示
                plt.clf()
                plt.xlabel('x')
                plt.ylabel('y')
                plt.plot(citys[..., 0], citys[..., 1], 'ob', ms=3)
                plt.plot(citys[best_path, 0], citys[best_path, 1])
                plt.plot([citys[best_path[-1], 0], citys[best_path[0], 0]],
                         [citys[best_path[-1], 1], citys[best_path[0], 1]], ms=2)
                ax = plt.gca()

                plt.text(0.1, 0.9, "distance:%.3f " % min_distance, transform=ax.transAxes)
                plt.pause(0.2)
            #            plt.text(0.1, 0.8, "T:%.3f °C" % T0, transform=ax.transAxes)
            value.append(min_distance)

        # print(best_path)
        # print(min_distance)
        if (p >= N):
            T0 = alpha * T0
        T0 = alpha * T0  # 更新温度
        print("temperature:%f" % T0)
    return best_path, min_distance


def sa(citys, tmp_path, tmp_distance, distance):
    # 初始化参数
    iteration1 = 10000  # 外循环迭代次数
    T0 = 1000  # 初始温度，取大些
    Tf = 1e-3  # 截止温度，可以不用
    alpha = 0.99  # 温度更新因子
    iteration2 = 200  # 内循环迭代次数
    min_distance = tmp_distance  # 最佳距离
    best_path = tmp_path
    value = []
    now_value = []  # 当次最优解

    num_itera = 0  # 记录迭代次数
    M = 0  # 记录进入if的次数
    N = 188  # 能量标准值
    plt.ion()
    while (T0 > Tf):
        p = 0
        for k in range(iteration2):
            M = M + 1

            # 生成新解
            # x1, s = updataPath(tmp_path)
            x1,s = updataPath2(tmp_path,tmp_distance,T0,value,min_distance)
            # now_value.append(s)
            # 判断是否更新解
            if s <= tmp_distance:  # 如果小于 直接接受
                M = 0
                tmp_distance = s
                tmp_path = x1.copy()
                num_itera += 1
                p = p + 1
            if s > tmp_distance:  # 如果大于 以一定概率接受
                deltaf = s - tmp_distance
                if random() < exp(-deltaf / T0):
                    M = 0
                    tmp_distance = s
                    tmp_path = x1.copy()
                    num_itera += 1
            if s < min_distance:  # 记录最优解
                min_distance = s
                best_path = x1.copy()
                #动画演示
                plt.clf()
                plt.xlabel('x')
                plt.ylabel('y')
                plt.plot(citys[..., 0], citys[..., 1], 'ob', ms=3)
                plt.plot(citys[best_path, 0], citys[best_path, 1])
                plt.plot([citys[best_path[-1], 0], citys[best_path[0], 0]],
                         [citys[best_path[-1], 1], citys[best_path[0], 1]], ms=2)
                ax = plt.gca()

                plt.text(0.1, 0.9, "distance:%.3f " % min_distance, transform=ax.transAxes)
                plt.pause(0.2)
            # plt.text(0.1, 0.8, "T:%.3f °C" % T0, transform=ax.transAxes)
            value.append(min_distance)


            # 如何一直没有进入if ，进行重升温，提高其进入if的概率
            # if (M > 100):
            #     tmp_path = best_path  # 将当前最优路径作为tmp
            #     M = 0
            #     T0 = T0 / 0.95  # 升温

        # print(best_path)
        # print(min_distance)
        if (p >= N):
             T0 = alpha * T0
             x1, s = updataPath2(tmp_path, tmp_distance, T0, value, min_distance)
             while(s >= tmp_distance)
                 x1, s = updataPath2(tmp_path, tmp_distance, T0, value, min_distance)
            tmp_distance = s
            tmp_path = x1.copy()
        T0 = alpha * T0  # 更新温度
        print("temperature:%f" % T0)
        # plt.text(0, 5000, "T:%.3f °C" % T0)
        # plt.text(0, 4800, "distance:%.3f " % min_distance)

    # 二次退火
    best_path,min_distance = sa_again(best_path,min_distance,value)

    # best_path, min_distance = sa_small_change(best_path,min_distance,value)

    print(best_path)
    print(min_distance)

    plt.xlabel('x')
    plt.ylabel('y')
    plt.plot(citys[..., 0], citys[..., 1], 'ob', ms=3)
    plt.plot(citys[best_path, 0], citys[best_path, 1])
    plt.plot([citys[best_path[-1], 0], citys[best_path[0], 0]],
             [citys[best_path[-1], 1], citys[best_path[0], 1]], ms=2)
    ax = plt.gca()
    plt.text(0.1,0.8,"T:%.3f °C"%T0, transform = ax.transAxes)
    plt.text(0.1,0.9,"distance:%.3f " % min_distance,transform = ax.transAxes)
    plt.ioff()
    plt.show()

    plt.plot(value)
    plt.xlabel("iteration")
    # plt.plot(now_value)
    plt.show()


if __name__ == '__main__':
    citys, distance = getCityData()
    start_path, start_distance = init(distance)
    sa(citys, start_path, start_distance, distance)
# 计算距离矩阵
