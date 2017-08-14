#! -*- coding=utf-8 -*-

# 面积法
def way1():
    import pylab as pl
    from math import log,exp,sqrt

    evaluate_result="you file path"
    db = [] #[score,nonclk,clk]
    pos, neg = 0, 0
    with open(evaluate_result,'r') as fs:
     for line in fs:
     nonclk,clk,score = line.strip().split('\t')
     nonclk = int(nonclk)
     clk = int(clk)
     score = float(score)
     db.append([score,nonclk,clk])
     pos += clk
     neg += nonclk


    db = sorted(db, key=lambda x:x[0], reverse=True)

    #计算ROC坐标点
    xy_arr = []
    tp, fp = 0., 0. 
    for i in range(len(db)):
     tp += db[i][2]
     fp += db[i][3]
     xy_arr.append([fp/neg,tp/pos])

    #计算曲线下面积
    auc = 0. 
    prev_x = 0
    for x,y in xy_arr:
     if x != prev_x:
     auc += (x - prev_x) * y
     prev_x = x

    print "the auc is %s."%auc

    x = [_v[0] for _v in xy_arr]
    y = [_v[1] for _v in xy_arr]
    pl.title("ROC curve of %s (AUC = %.4f)" % ('svm',auc))
    pl.xlabel("False Positive Rate")
    pl.ylabel("True Positive Rate")
    pl.plot(x, y)# use pylab to plot x and y
    pl.show()# show the plot on the screen

# sklearn接口
def way2():
    from sklearn.metrics import roc_auc_score
    score = [1.1,0.2,3.4,5.1,2.1,0.4,2.4]
    label = [0,0,1,0,1,0,0]
    auc = roc_auc_score(label,score)
    print 'AUC:',auc

# Rank评分法
def way3():
  def getAuc(labels, pred):
      '''将pred数组的索引值按照pred[i]的大小正序排序，返回的sorted_pred是一个新的数组，
         sorted_pred[0]就是pred[i]中值最小的i的值，对于这个例子，sorted_pred[0]=8
      '''
      sorted_pred = sorted(range(len(pred)), key=lambda i: pred[i])
      pos = 0.0  # 正样本个数
      neg = 0.0  # 负样本个数
      auc = 0.0
      last_pre = pred[sorted_pred[0]]
      count = 0.0
      pre_sum = 0.0  # 当前位置之前的预测值相等的rank之和，rank是从1开始的，所以在下面的代码中就是i+1
      pos_count = 0.0  # 记录预测值相等的样本中标签是正的样本的个数
      for i in range(len(sorted_pred)):
          if labels[sorted_pred[i]] > 0:
              pos += 1
          else:
              neg += 1
          if last_pre != pred[sorted_pred[i]]:  # 当前的预测概率值与前一个值不相同
              # 对于预测值相等的样本rank需要取平均值，并且对rank求和
              auc += pos_count * pre_sum / count
              count = 1
              pre_sum = i + 1  # 更新为当前的rank
              last_pre = pred[sorted_pred[i]]
              if labels[sorted_pred[i]] > 0:
                  pos_count = 1  # 如果当前样本是正样本 ，则置为1
              else:
                  pos_count = 0  # 反之置为0
          else:
              pre_sum += i + 1  # 记录rank的和
              count += 1  # 记录rank和对应的样本数，pre_sum / count就是平均值了
              if labels[sorted_pred[i]] > 0:  # 如果是正样本
                  pos_count += 1  # 正样本数加1
      auc += pos_count * pre_sum / count  # 加上最后一个预测值相同的样本组
      auc -= pos * (pos + 1) / 2  # 减去正样本在正样本之前的情况
      auc = auc / (pos * neg)  # 除以总的组合数
      return auc
