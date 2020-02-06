<center><b><font size="5">基于看山杯的用户回答率预测算法</font></b></center>

【队伍名】“不拿前三也没法”团队  
【队员及机构】  
成员：钟睿、方齐昱、邓皓、张梧桐、陈伟  
机构：北京邮电大学  
【摘要】  
本报告是关于智源-看山杯2019专家发现算法大赛第五名的解决方案。在知识分享或问答社区中，问题数远远超过有质量的回复数。因此，如何连接知识、专家和用户，增加专家的回答意愿，成为了此类服务的中心课题。该比赛提供了知乎的问题信息、用户画像、用户回答记录，以及用户接受邀请的记录，要求选手预测这个用户是否会接受某个新问题的邀请，回答问题。我们在该比赛中对数据进行深度的特征挖掘，生成统计类特征、图特征、距离特征等，经多种方案试错，采用多模型训练融合的方式，取得了第五名的成绩。  
【关键词】LightGBM、神经网络、XdeepFM、模型融合、专家发现算法  
## 数据探索  
知乎提供了数据问题信息（问题表）、用户画像（用户表）、用户回答记录（回答表），为了保护用户隐私, 所有的用户ID、问题ID、回答ID、话题ID 都经过特殊编码处理，所有问题的标题、问题的描述、回答的正文以及话题名称这些原始信息都相应的被编码成单字 ID 序列和词语 ID 序列，所有问题、回答以及邀请的时间均进行了相应的偏移，偏移后的时间给出到小时的精度, 日期格式为 D1-H3, 含义为 day 1 的 3 点。我们需要将具有标注信息的邀请数据（邀请表）当作训练集，预测专家接受问题邀请的概率。使用 AUC 对参赛队伍提交的数据与真实的数据进行衡量评估：
![](https://user-images.githubusercontent.com/25412051/72258269-9b57f700-3648-11ea-90a3-6798f3489bae.png)
AUC计算方式如上式所示，M为正样本数，N为负样本数，ranki为第i个正样本所在的位置。
首先，我们对日期格式进行处理，转换为小时，观察训练集和A版测试集在小时粒度上的时间分布，如图1-a。
![](https://tva1.sinaimg.cn/large/006tNbRwly1gav37ournkj30bc06n402.jpg)
<center>图1-a</center>
根据回答表，查看用户回答问题在小时粒度上的时间分布（如图1-b），24小时的时间分布（如图1-c），周一到周日的时间分布（如图1-d），以天为粒度的时间分布（如图1-e）。
![](https://tva1.sinaimg.cn/large/006tNbRwly1gav386359rj30bc04xdhe.jpg)
<center>图1-b</center>
![](https://tva1.sinaimg.cn/large/006tNbRwly1gav38fam8mj30bc05d0t4.jpg)
<center>图1-c</center>
![](https://tva1.sinaimg.cn/large/006tNbRwly1gav38n01fhj30bc05b3yr.jpg)
<center>图1-d</center>
![](https://tva1.sinaimg.cn/large/006tNbRwly1gav38w3u4jj30bc059mxn.jpg)
<center>图1-e</center>
根据邀请表，统计用户接受邀请率在小时粒度上的时间分布（如图1-f），24小时的时间分布（如图1-g），周一到周日的时间分布（如图1-h），以天为粒度的时间分布（如图1-i）。
![](https://tva1.sinaimg.cn/large/006tNbRwly1gav392yb20j30bc05kmym.jpg)
<center>图1-f</center>
![](https://tva1.sinaimg.cn/large/006tNbRwly1gav3997qcpj30bc05bmxt.jpg)
<center>图1-g</center>
![](https://tva1.sinaimg.cn/large/006tNbRwly1gav39hhjpkj30bc05ft93.jpg)
<center>图1-h</center>
![](https://tva1.sinaimg.cn/large/006tNbRwly1gav39nz73qj30bc05ddgi.jpg)
<center>图1-i</center>
根据图1-a可以看出线上测试集数据时间在训练集之后，每小时的数据量大概占训练集的一半左右，可以由此估测A版、B版测试集是同一时间段内的随机采样。由回答和用户接受邀请率的时间分布可知，两者和时间具有一定的相关性，并且在24小时的分布上有显著变化。
## 解决思路
### 特征生成
由于该比赛数据有较强的时间相关性，在特征生成时需注意数据穿越问题，比如计算某条邀请相关的用户历史特征时，应该根据该邀请时间点之前的数据生成。以下介绍生成的特征：
#### 基本特征
对知乎提供的原始特征进行处理，针对离散特征进行编码或者one-hot，丢弃只有单一值的特征。
#### topic（主题）聚类
对于topic向量进行K-means聚类，从数万级减少到了百级，然后将每簇的中心的topic向量作为该簇的向量。以下所提及的topic向量皆为聚类后的向量。
#### 统计类特征
根据数据，生成问题的统计类特征，如问题的全部邀请数，问题的历史受邀请数，问题邀请的轮次，问题邀请的频率，问题的历史答案特征等。
根据数据，生成用户的统计类特征，如用户的全部受邀请数，用户的历史受邀请数，用户的历史接受邀请数，用户的历史回答率，用户上次接受邀请距今多久，用户的历史回答特征等。
统计全部被回答问题的词的出现频次，对每个问题的词频次取平均、最大值、最小值作为该问题的特征之一。
#### 图特征
将用户id、问题id和主题id作为节点，创建图结构。用户id和问题id的边是回答关系，用户id和主题id是指用户感兴趣和关注的主题，问题id和主题id的边是指问题对应的主题。图特征计算的是用户id节点到问题id节点的路径模式的个数以及问题和用户历史回答过的问题的相似性。
#### 问题的embedding特征
将问题的词向量做加权平均，当作问题的句向量，以表征该问题。
#### 距离特征
通过多种距离算法计算用户感兴趣的topic和问题的topic的距离特征。
计算当前邀请的问题和用户历史回答的相关距离。如当前邀请的问题和用户历史回答问题标题的距离，回答内容的距离等。
根据回答和主题，将问题id、用户id、主题id一起放入一个图中，利用随机游走构建路径，然后用word2vec来生成低维向量，再计算问题id和用户id的距离。
### 训练模型
下面介绍本方案使用的相关模型。
#### LightGBM
##### 模型概述
LightGBM 是一个梯度boosting框架, 使用基于学习算法的决策树. 它是分布式的, 高效的，它具有速度和内存使用的优化、稀疏优化、准确率的优化、网络通信的优化、并行学习的优化、GPU 支持可处理大规模数据等方面的优势。
因为LightGBM的高效性，我们将LightGBM作为主要模型，同时也将其用于对有效特征的快速验证。
##### LightGBM特征介绍
在本方案中，前述的基本特征、统计类特征、图特征、问题的embedding特征、距离特征等231维特征皆可直接输入到LightGBM中。
#### 神经网络模型
##### 模型概述
神经网络可以通过拟合高阶的非线性关系，同时减少了人工特征的工作量，同时能使用更多的离散特征并进行特征交叉，提升模型的泛化能力，我们使用的神经网络模型在A榜的最高得分是0.8623。
目前推荐系统领域主流的深度学习模型有Wide&Deep，DeepFM，XdeepFM[2]等。此次比赛这些模型我们都有尝试，DeepFM较Wide&Deep模型可以省去手动特征交叉，XdeepFM较DeepFM模型进一步增加了压缩交互网络（CIN）如图2-a，它考虑了以下几个方面：
![](https://tva1.sinaimg.cn/large/006tNbRwly1gav39unbd7j3097074751.jpg)
<center>图2-a</center>
（1）交互是在向量层次上应用的，而不是在位层次上；  
（2）高阶特征交互是明确测量的；  
（3）网络的复杂度不会随着相互作用的程度增加。  
因此最终我们采用了XdeepFM模型中的CIN模块进行特征交叉。  
其次，我们还考虑对用户的历史行为序列进行建模。我们将用户历史序列特征输入LSTM网络中，得到的输出结果与其他特征一起进入CIN模块进行特征交叉。  
网络模型结构如下图2-b所示：
![](https://tva1.sinaimg.cn/large/006tNbRwly1gav3a07gaij30bc05h0tu.jpg)
<center>图2-b</center>
##### NN特征处理介绍
（1）连续特征  
前述的大部分基本特征、统计类特征、图特征、距离特征等都属于连续特征。  
我们尝试了三种方法将连续特征输入到网络：  
    （a）归一化后直接输入dnn部分，不参与特征交叉  
    （b）离散化后再embedding，他离散特征的embedding一起参与特征交叉  
    （c）为每一个field下的连续特征维护一个embedding vector  a,取 a*v作为其最终的embedding表示，与其他离散特征的embedding一起参与特征交叉  
最终采用方案（c） 

（2）离散特征  
低频过滤：离散特征中，有些类别的取值比较低频，存在分布不均的情况，因此我们将这部分特征进行了低频过滤处理，将出现频次小于阈值的值归为一类。  
Embedding：对于一些高维稀疏的id类特征，例如用户，问题id等，可以作为deep模型的embedding输入，通过embedding层转换成低维且稠密的向量。降维后的特征具有较强的“扩展能力”，能够探索历史数据中从未出现过的特征组合，增强模型的表达能力。  
（3）序列特征  
我们抽取了用户的历史回答问题记录，通过LSTM模型对其短期兴趣进行建模。即将用户近3次回答的历史回答问题的题目词向量输入到LSTM中得到embedding向量，与其他特征一起输入到NN网络中。  
### 其他尝试  
#### 细粒度交叉特征  
主要是针对LightGBM模型的输入，将比较重要的特征进行手动特征交叉。例如将创建时间的week(取值0~6)，创建时间的hour(取值0~23),进行手动交叉，生成取值0~167的细粒度离散特征，并统计其对应的回答率作为统计特征。如下图2-c为采用expanding meaning方法构造的回答率统计特征，可以看到，将创建时间hour和week手动交叉后，再构造统计特征更具有相比较原来的特征分布更加平滑并且更具有区分度。此类特征在低分段82分以下时，线上会有提分，高分段时则线上没有提分。  
![](https://tva1.sinaimg.cn/large/006tNbRwly1gav3a6p9sgj30970ajwfm.jpg)
<center>图2-c</center>
#### LightGBM+DeepFM模型  
由于我们的特征中含有大量的统计类历史特征，这些特征存在不同程度的缺失，直接输入DeepFm效果并不好，因此将其先输入LightGBM,并将其叶子节点和其他id类型的特征一起输入到DeepFM模型中。  
使用LightGBM来进行可以起到对统计类特征的离散化、缺失值填充和约束embedding空间的作用。所用的模型结构如下图2-d所示：  
![](https://tva1.sinaimg.cn/large/006tNbRwly1gav3ac3ft8j30b4068t9j.jpg)
<center>图2-d Lightgbm+DeepFm模型</center>
但在实际操作中，若将叶子节点作为中间结果保存时，会产生150G以上的中间特征，结果太大，保存和读取都极为不方便，因此只保存Lightgbm模型，并在每一batch中从原始特征重新预测叶子节点（这个过程使用了多线程来优化）。  
最终，将统计特征输入到lightgbm ,并在进入embedding层前添离散特征，用户id,问题id等其他特征来作为DeepFm模型的输入。将其结果与Lightgbm模型结果融合后，在0.85分段线上可提升0.0035，但在0.86分以上时无效。    
## 总结  
这次比赛是知乎业务场景下的问题路由问题。首先与其他传统CTR比赛相比，比赛数据中多了许多和文本内容相关的特征。但我们许多关于文本内容相关的特征构造方法都没有拿到很好的效果，这一点非常遗憾。其次，比赛过程中我们的LGB模型的效果一直优于NN模型，但是在官方讲座中得知在工业应用里是相反的，有可能是因为我们的大部分特征都是连续型特征，没有更好的利用到id类型的特征。最后，我们将LGB模型和NN模型的预测结果以算术平均的方式做模型融合，达到了线上0.88721的分数，取得了第五名，而第一名的分数为0.89697。  

## 参考文献  
[1] Ke G, Meng Q, Finley T, et al. Lightgbm: A highly efficient gradient boosting decision tree[C]//Advances in Neural Information Processing Systems. 2017: 3146-3154.  
[2] Lian J, Zhou X, Zhang F, et al. xdeepfm: Combining explicit and implicit feature interactions for recommender systems[C]//Proceedings of the 24th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining.   ACM, 2018: 1754-1763.  
[3] He X, Pan J, Jin O, et al. Practical lessons from predicting clicks on ads at facebook[C]//Proceedings of the Eighth International Workshop on Data Mining for Online Advertising. ACM, 2014: 1-9.  
 




## EDA
知乎提供了数据问题信息（问题表）、用户画像（用户表）、用户回答记录（回答表），为了保护用户隐私, 所有的用户ID、问题ID、回答ID、话题ID 都经过特殊编码处理，所有问题的标题、问题的描述、回答的正文以及话题名称这些原始信息都相应的被编码成单字 ID 序列和词语 ID 序列，所有问题、回答以及邀请的时间均进行了相应的偏移，偏移后的时间给出到小时的精度, 日期格式为 D1-H3, 含义为 day 1 的 3 点。我们需要将具有标注信息的邀请数据（邀请表）当作训练集，预测专家接受问题邀请的概率。下面对数据进行一定地分析：

加载数据


```python
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm_notebook as tqdm
```


```python
folder_path = '../data/origin/'
config_single_word =['SW_id','SW_embeding']
config_word =['W_id','W_embeding']
config_topic =['T_id','T_embeding']
config_question = ['Q_id','Q_Create_Time','QTitle_SW_list','QTitle_W_list',
                   'QDescrible_SW_list','QDescrible_W_list','QT_list']
config_answer = ['A_id','Q_id','M_id','A_Create_Time','A_SW_list','A_W_list',
                 'IsGood','IsRecommend','IsAccept','IncludePicture','IncludeVideo',
                 'Count_SW','Count_Like','Count_CancelLike','Count_Comment',
                 'Count_Collect','Count_Thank','Count_Report','Count_noHelp','Count_Oppose']
config_member = ['M_id','Gender','Create_KeyW','Create_Count','Create_hot','Reg_Type',
                 'Reg_Platform','Frequency','Feature_A','Feature_B','Feature_C','Feature_D','Feature_E',
                 'Feature_a','Feature_b','Feature_c','Feature_d','Feature_e',
                 'MemberScore','T_attention','T_interest']
config_invite =['Q_id','M_id','Invite_Time','Label']
config_invite_test =['Q_id','M_id','Invite_Time']
print(os.listdir(folder_path))
answer_info = pd.read_csv(f'{folder_path}answer_info_0926.txt',
                          sep='\t',header=None,names=config_answer)
invite_info = pd.read_csv(f'{folder_path}invite_info_0926.txt',
                          sep='\t',header=None,names=config_invite)
member_info = pd.read_csv(f'{folder_path}member_info_0926.txt',
                          sep='\t',header=None,names=config_member)
question_info = pd.read_csv(f'{folder_path}question_info_0926.txt',
                            sep='\t',header=None,names=config_question)
test_data = pd.read_csv(f'{folder_path}invite_info_evaluate_1_0926.txt',
                                     sep='\t',header=None,names=config_invite_test)
print('end!')
```

    ['invite_info_evaluate_1_0926.txt', 'topic_vectors_64d.txt', 'word_vectors_64d.txt', 'member_info_0926.txt', 'invite_info_0926.txt', 'question_info_0926.txt', 'single_word_vectors_64d.txt', 'answer_info_0926.txt']
    end!
    


```python
def DH2Hour(time:str):
    t=time.split("-")
    return int(t[0][1:])*24+int(t[1][1:])
def HourOfDay(hour:int):
    return int(hour%24)
def DayOfWeek(hour:int):
    return int(hour/24%7)
def DayOfMonth(hour:int):
    return int(hour/24%30)
answer_info['A_Create_Time_Hour']=answer_info['A_Create_Time'].apply(lambda x:DH2Hour(x))
invite_info['Invite_Time_Hour']=invite_info['Invite_Time'].apply(lambda x:DH2Hour(x))
question_info['Q_Create_Time_Hour']=question_info['Q_Create_Time'].apply(lambda x:DH2Hour(x))
test_data['Invite_Time_Hour']=test_data['Invite_Time'].apply(lambda x:DH2Hour(x))
invite_info['AcceptRateByHour']=invite_info.groupby(['Invite_Time_Hour'])['Label'].transform('sum')/invite_info.groupby(['Invite_Time_Hour'])['Label'].transform('count')

answer_info=answer_info.sort_values('A_Create_Time_Hour')
question_info=question_info.sort_values('Q_Create_Time_Hour')
member_info=member_info.drop_duplicates(subset=['M_id'])
invite_info=invite_info.sort_values('Invite_Time_Hour').drop_duplicates()
test_data=test_data.sort_values('Invite_Time_Hour').drop_duplicates()

for func in tqdm([HourOfDay,DayOfWeek,DayOfMonth],desc='HourOfDay_DayOfWeek_DayOfMonth'):
    answer_info[f'A_Create_Time_{func.__name__}']=answer_info['A_Create_Time_Hour'].apply(lambda x:func(x))
    invite_info[f'Invite_Time_{func.__name__}']=invite_info['Invite_Time_Hour'].apply(lambda x:func(x))
    question_info[f'Q_Create_Time_{func.__name__}']=question_info['Q_Create_Time_Hour'].apply(lambda x:func(x))
    test_data[f'Invite_Time_{func.__name__}']=test_data['Invite_Time_Hour'].apply(lambda x:func(x))
    invite_info[f'AcceptRateBy{func.__name__}']=invite_info.groupby([f'Invite_Time_{func.__name__}'])['Label'].transform('sum')/invite_info.groupby([f'Invite_Time_{func.__name__}'])['Label'].transform('count')
    test_data=pd.merge(test_data,
                        invite_info[[f'Invite_Time_{func.__name__}',f'AcceptRateBy{func.__name__}']].drop_duplicates(),
                        on=[f'Invite_Time_{func.__name__}'], how='left')
print('end!')
```


<p>Failed to display Jupyter Widget of type <code>HBox</code>.</p>
<p>
  If you're reading this message in the Jupyter Notebook or JupyterLab Notebook, it may mean
  that the widgets JavaScript is still loading. If this message persists, it
  likely means that the widgets JavaScript library is either not installed or
  not enabled. See the <a href="https://ipywidgets.readthedocs.io/en/stable/user_install.html">Jupyter
  Widgets Documentation</a> for setup instructions.
</p>
<p>
  If you're reading this message in another frontend (for example, a static
  rendering on GitHub or <a href="https://nbviewer.jupyter.org/">NBViewer</a>),
  it may mean that your frontend doesn't currently support widgets.
</p>



    
    end!
    

首先，我们对日期格式进行处理，转换为小时，观察训练集和A版测试集在小时粒度上的时间分布，如下图。


```python
start=invite_info['Invite_Time_Hour'].min()
plt.hist(invite_info['Invite_Time_Hour']-start, label='train',
         bins=invite_info['Invite_Time_Hour'].max()-invite_info['Invite_Time_Hour'].min()+1)
plt.hist(test_data['Invite_Time_Hour']-start, label='test',
         bins=test_data['Invite_Time_Hour'].max()-test_data['Invite_Time_Hour'].min())
plt.legend()
plt.title('Distribution of invite time hour')
```




    Text(0.5,1,'Distribution of invite time hour')




![png](output_7_1.png)


根据回答表  
查看用户回答问题在小时粒度上的时间分布（如图Distribution of Answer Create Time hour）  
24小时的时间分布（如图Distribution of Answer Create Time HourOfDay）  
周一到周日的时间分布（如图Distribution of Answer Create Time DayOfWeek）  
以天为粒度的时间分布（如图Distribution of Answer Create Time DayOfMonth）  


```python
plt.subplots_adjust(left = 1,right = 4,bottom=1,top=3)
plt.subplot(2,2,1)
plt.hist(answer_info['A_Create_Time_Hour']-answer_info['A_Create_Time_Hour'].min(),
         bins=answer_info['A_Create_Time_Hour'].max()-answer_info['A_Create_Time_Hour'].min()+1)
plt.title('Distribution of Answer Create Time hour')

plt.subplot(2,2,2)
plt.hist(answer_info['A_Create_Time_HourOfDay'],
         bins=answer_info['A_Create_Time_HourOfDay'].max()-answer_info['A_Create_Time_HourOfDay'].min()+1)
plt.title('Distribution of Answer Create Time HourOfDay')

plt.subplot(2,2,3)
plt.hist(answer_info['A_Create_Time_DayOfWeek'],
         bins=answer_info['A_Create_Time_DayOfWeek'].max()-answer_info['A_Create_Time_DayOfWeek'].min()+1)
plt.title('Distribution of Answer Create Time DayOfWeek')

plt.subplot(2,2,4)
plt.hist(answer_info['A_Create_Time_DayOfMonth'],
         bins=answer_info['A_Create_Time_DayOfMonth'].max()-answer_info['A_Create_Time_DayOfMonth'].min()+1)
plt.title('Distribution of Answer Create Time DayOfMonth')
plt.show()
```


![png](output_9_0.png)


根据邀请表  
统计用户接受邀请率在小时粒度上的时间分布（如图Distribution of Invite Time Hour）  
24小时的时间分布（如图Distribution of Invite Time HourOfDay）  
周一到周日的时间分布（如图Distribution of Invite Time DayOfWeek）  
以天为粒度的时间分布（如图Distribution of Invite Time DayOfMonth）  


```python
plt.subplots_adjust(left = 1,right = 4,bottom=1,top=3)
plt.subplot(2,2,1)
x,y="Invite_Time_Hour","AcceptRateByHour"
data=invite_info[[x,y]].drop_duplicates()
data[x]=data[x]-data[x].min()
sns.barplot(x=x, y=y, data=data)
plt.title('Distribution of Invite Time Hour')
plt.subplot(2,2,2)
x,y="Invite_Time_HourOfDay","AcceptRateByHourOfDay"
sns.barplot(x=x, y=y, data=invite_info[[x,y]].drop_duplicates()) 
plt.title('Distribution of Invite Time HourOfDay')
plt.subplot(2,2,3)
x,y="Invite_Time_DayOfWeek","AcceptRateByDayOfWeek"
sns.barplot(x=x, y=y, data=invite_info[[x,y]].drop_duplicates())
plt.title('Distribution of Invite Time DayOfWeek')
plt.subplot(2,2,4)
x,y="Invite_Time_DayOfMonth","AcceptRateByDayOfMonth"
sns.barplot(x=x, y=y, data=invite_info[[x,y]].drop_duplicates())
plt.title('Distribution of Invite Time DayOfMonth')
plt.show()
```


![png](output_11_0.png)


根据图Distribution of invite time hour可以看出线上测试集数据时间在训练集之后，覆盖的时间段大概是训练集的四分之一，每小时的数据量大概占训练集的一半左右，可以由此估测A版、B版测试集是同一时间段内的随机采样。由回答和用户接受邀请率的时间分布可知，两者和时间具有一定的相关性，并且在24小时的分布上有显著变化。

## 特征工程

### 统计类及表征类特征
    DataReader用于读取各类数据集，FeatureExtractor用于抽取统计类特征。
    统计类特征包括了：
    1.问题的基本信息类特征
    2.用户的基本信息类特征
    3.问题的受邀特征
    4.用户的受邀特征
    5.样本受邀时间特征
    6.用户历史回答特征
    7.用户转化率特征
    8.用户和问题的密度特征
    9.用户和问题的topic距离特征


```python
config_single_word =['SW_id','SW_embeding']
config_word =['W_id','W_embeding']
config_topic =['T_id','T_embeding']
config_question = ['Q_id','create_time','QTitle_SW_list','QTitle_W_list','QDescrible_SW_list','QDescrible_W_list','QT_list']
config_answer = ['A_id','Q_id','M_id','A_create_time','A_SW_list','A_W_list','IsGood','IsRecommend','IsAccept','IncludePicture','IncludeVideo','Count_SW','Count_Like','Count_CancelLike','Count_comment','Count_Collect','Count_Thank','Count_Report','Count_noHelp','Count_Oppose']
config_member = ['M_id','Gender','Create_KeyW','Create_Count','Create_hot','Reg_Type','Reg_Platform','Frequency','Feature_A','Feature_B','Feature_C','Feature_D','Feature_E','Feature_a','Feature_b','Feature_c','Feature_d','Feature_e','Score','T_attention','T_interest']
config_invite =['Q_id','M_id','Invite_Time','Label',"Folder"]
config_invite_test =['Q_id','M_id','Invite_Time']

FILE_PATH = "../data/origin/"


from tqdm import tqdm
import pandas as pd
import numpy as np
import json
from itertools import chain


class DataReader(object):
    """
    读取数据的生成器
    """
    def __new__(cls, file_name):
        return getattr(cls, file_name)

    def single_word_vectors_64d(desc=""):
        with open(FILE_PATH + "single_word_vectors_64d.txt") as file:
            for line in tqdm(file, total=2*10**4, desc=desc, mininterval=1):
                value_list = line.strip("\n").split("\t")
                key_list = config_single_word
                yield dict(zip(key_list, value_list))
                
    def word_vectors_64d(desc=""):
        with open(FILE_PATH + "word_vectors_64d.txt") as file:
            for line in tqdm(file, total=176*10**4, desc=desc, mininterval=1):
                value_list = line.strip("\n").split("\t")
                key_list = config_word
                yield dict(zip(key_list, value_list))

    def topic_vectors_64d(desc=""):
        with open(FILE_PATH + "topic_vectors_64d.txt") as file:
            for line in tqdm(file, total=10*10**4, desc=desc, mininterval=1):
                value_list = line.strip("\n").split("\t")
                key_list = config_topic
                yield dict(zip(key_list, value_list))

    def member_info(desc=""):
        with open(FILE_PATH + "member_info_0926.txt") as file:
            for line in tqdm(file, total=1931654, desc=desc, mininterval=1):
                value_list = line.strip("\n").split("\t")
                key_list = config_member
                yield dict(zip(key_list, value_list))

    def question_info(desc=""):
        with open(FILE_PATH + "question_info_0926.txt") as file:
            for line in tqdm(file, total=1829900, desc=desc, mininterval=1):
                value_list = line.strip("\n").split("\t")
                key_list = config_question
                yield dict(zip(key_list, value_list))

    def answer_info(desc=""):
        with open(FILE_PATH + "answer_info_0926.txt") as file:
            for line in tqdm(file, total=4513735, desc=desc, mininterval=1):
                value_list = line.strip("\n").split("\t")
                key_list = config_answer
                yield dict(zip(key_list, value_list))

    def invite_info(desc=""):
        with open("../data/train/train_data.txt") as file:
            next(file)
            for line in tqdm(file, total=9489162, desc=desc, mininterval=1):
                value_list = line.strip("\n").split(",")
                key_list = config_invite
                yield dict(zip(key_list, value_list))

    def invite_info_evaluate(desc=""):
        with open("../data/test/test_data.txt") as file:
            next(file)
            for line in tqdm(file, total=1141682, desc=desc, mininterval=1):
                value_list = line.strip("\n").split(",")
                key_list = config_invite_test
                yield dict(zip(key_list, value_list))
                
    def invite_info_final(desc=""):
        with open("../data/final/test.txt") as file:
            next(file)
            for line in tqdm(file, total=1141718, desc=desc, mininterval=1):
                value_list = line.strip("\n").split(",")
                key_list = config_invite_test
                yield dict(zip(key_list, value_list))
                
                
class FeatureExtractor(object):
    """
    特征抽取器
    """
    def __init__(self, train_or_test):
        if train_or_test not in ["train", "test", "final"]:
            raise Exception("[train_or_test value is wrong]")
            
        self.train_or_test = train_or_test
        self.save_path = "features/" + train_or_test + "/"
        
        self.single_word_vectors = DataReader("single_word_vectors_64d")
        self.word_vectors = DataReader("word_vectors_64d")
        self.topic_vectors = DataReader("topic_vectors_64d")
        self.question_info = DataReader("question_info")
        self.member_info = DataReader("member_info")
        self.answer_info = DataReader("answer_info")
        
        if train_or_test == "train":
            self.invite_info = DataReader("invite_info") 
        elif train_or_test == "test":
            self.invite_info = DataReader("invite_info_evaluate")
        elif train_or_test == "final":
            self.invite_info = DataReader("invite_info_final")
        else:
            raise

    def extractInvitedTime(self):
        """
        问题受邀时间特征
        date_invite2create: 该问题受邀时间距创建时间的天数
        invite_week：       受邀日期是周几（7维独热码）
        invite_hour：       受邀时刻（24维独热码）
        cur_hour_answer_rate: 当前邀请小时的回答率
        cur_week_answer_rate: 当前邀请周的回答率
        """
        hour2answer_rate = {0: 0.2058, 1: 0.2375, 2: 0.2891, 3: 0.2549, 4: 0.219, 5: 0.3027, 6: 0.2461, 7: 0.1908, 8: 0.1623, 9: 0.1936, 10: 0.1887, 11: 0.177, 12: 0.1769, 13: 0.1835, 14: 0.1861, 15: 0.17, 16: 0.1713, 17: 0.1743, 18: 0.1851, 19: 0.1458, 20: 0.1764, 21: 0.1579, 22: 0.1514, 23: 0.1589}
        week2answer_rate = {0: 0.1831, 1: 0.1671, 2: 0.1815, 3: 0.1859, 4: 0.1812, 5: 0.1648, 6: 0.1862}
        question_table = {}
        for data in self.question_info("extractInvitedTime-1/2"):
            qid = data["Q_id"]
            date = int(data["create_time"].split("-")[0][1:])
            hour = int(data["create_time"].split("-")[1][1:])
            if qid not in question_table or (date, hour) < question_table[qid]:
                question_table[qid] = (date, hour)
        
        date_invite2create = []
        invite_week = [[],[],[],[],[],[],[]]
        invite_hour = [[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[]]
        cur_hour_answer_rate = []
        cur_week_answer_rate = []
        for data in self.invite_info("extractInvitedTime-2/2"):
            qid = data["Q_id"]
            date = int(data["Invite_Time"].split("-")[0][1:])
            hour = int(data["Invite_Time"].split("-")[1][1:])
            week = date % 7
            
            span = (date-question_table[qid][0])*24+(hour-question_table[qid][1]) 
            date_invite2create.append(span if span >=0 else np.NaN)
            cur_hour_answer_rate.append(hour2answer_rate[hour])
            cur_week_answer_rate.append(week2answer_rate[week])
            for i, arr in enumerate(invite_week):
                arr.append(1 if i == week else 0)
            for i, arr in enumerate(invite_hour):
                arr.append(1 if i == hour else 0)
            
        np.save(self.save_path+"date_invite2create.npy", np.array([date_invite2create]))
        np.save(self.save_path+"invite_week.npy", np.array(invite_week))
        np.save(self.save_path+"invite_hour.npy", np.array(invite_hour))
        np.save(self.save_path+"cur_hour_answer_rate.npy", np.array([cur_hour_answer_rate]))
        np.save(self.save_path+"cur_week_answer_rate.npy", np.array([cur_week_answer_rate]))


    def extractQuestionInvite(self):
        """
        问题受邀特征
        q_inv_all: 问题的全部邀请数
        q_inv_before: 问题的历史受邀请数
        q_inv_round: 问题邀请的轮次
        q_inv_freq: 问题邀请的频率
        q_inv_round_freq: 问题轮次的邀请频率
        q_first_inv: 问题第一次邀请距今多久
        q_last_inv: 问题上一次邀请距今多久
        q_inv_time_span: 问题邀请的时间跨度
        q_inv_sametime_num: 问题同时邀请人数
        """
        record = {}
        for data in chain(DataReader("invite_info")("extractQuestionInvite-1/3"), 
                          DataReader("invite_info_evaluate")("extractQuestionInvite-2.5/3"), 
                          DataReader("invite_info_final")("extractQuestionInvite-2.5/3")):
            qid = data["Q_id"]
            invite_date = int(data["Invite_Time"].split("-")[0][1:])
            invite_time = int(data["Invite_Time"].split("-")[1][1:])
            if qid in record:
                record[qid] += [(invite_date, invite_time)]
            else:
                record[qid] = [(invite_date, invite_time)]

        for qid in tqdm(record):
            record[qid].sort()
        
        q_inv_all = []
        q_inv_before = []
        q_inv_round = []
        q_inv_freq = []
        q_inv_round_freq = []
        q_first_inv = []
        q_last_inv = []
        q_inv_time_span = []
        q_inv_sametime_num = []
        for data in self.invite_info("extractQuestionInvite-3/3"):
            qid = data["Q_id"]
            invite_date = int(data["Invite_Time"].split("-")[0][1:])
            invite_time = int(data["Invite_Time"].split("-")[1][1:])
            
            q_invite = [x for x in record[qid] if (invite_date-30, 0) < x < (invite_date, invite_time)]
            q_invite_round = [x for x,y in zip(record[qid], record[qid][1:]+[0]) if x!=y]
            
            first_inv = (invite_date-record[qid][0][0])*24+(invite_time-record[qid][0][1]) if record[qid] else np.NaN
            last_inv = (invite_date-q_invite[-1][0])*24+(invite_time-q_invite[-1][1]) if q_invite else np.NaN
            
            inv_freq = np.mean([(y[0]-x[0])*24+(y[1]-x[1]) for x,y in zip(record[qid], record[qid][1:])])  if len(record[qid])>1 else np.NaN
            round_freq = np.mean([(y[0]-x[0])*24+(y[1]-x[1]) for x,y in zip(q_invite_round, q_invite_round[1:])])  if len(q_invite_round)>1 else np.NaN

            q_inv_all.append(len(record[qid]))
            q_inv_before.append(len(q_invite))
            q_inv_round.append(len(q_invite_round))
            q_inv_freq.append(inv_freq)
            q_inv_round_freq.append(round_freq)
            q_first_inv.append(first_inv)
            q_last_inv.append(last_inv)
            q_inv_time_span.append(first_inv - last_inv if first_inv and last_inv else np.NaN)
            q_inv_sametime_num.append(len([x for x in record[qid] if x == (invite_date, invite_time)]))
            
        np.save(self.save_path+"q_inv_all.npy", np.array([q_inv_all]))
        np.save(self.save_path+"q_inv_before.npy", np.array([q_inv_before]))
        np.save(self.save_path+"q_inv_round.npy", np.array([q_inv_round]))
        np.save(self.save_path+"q_inv_freq.npy", np.array([q_inv_freq]))
        np.save(self.save_path+"q_inv_round_freq.npy", np.array([q_inv_round_freq]))
        np.save(self.save_path+"q_first_inv.npy", np.array([q_first_inv]))
        np.save(self.save_path+"q_last_inv.npy", np.array([q_last_inv]))
        np.save(self.save_path+"q_inv_time_span.npy", np.array([q_inv_time_span]))
        np.save(self.save_path+"q_inv_sametime_num.npy", np.array([q_inv_sametime_num]))

    def extractMemberInvite1(self):
        """
        用户受邀特征
        m_inv_all: 用户的全部受邀请数
        m_inv_before: 用户的历史受邀请数
        m_acpt_inv_before: 用户的历史接受邀请数
        m_acpt_rate_before: 用户的历史回答率
        m_last_acpt_inv: 用户上次接受邀请距今多久
        """
        record = {}
        for data in (DataReader("invite_info")("extractMemberInvite1-1/2")):
            mid = data["M_id"]
            invite_date = int(data["Invite_Time"].split("-")[0][1:])
            invite_hour = int(data["Invite_Time"].split("-")[1][1:])
            if mid in record:
                record[mid]["invite_cnt"] += [(invite_date, invite_hour)]
                record[mid]["pos_cnt"] += [(invite_date, invite_hour)] if data["Label"] == "1" else []
            else:
                record[mid] = {
                    "invite_cnt": [(invite_date, invite_hour)],
                    "pos_cnt": [(invite_date, invite_hour)] if data["Label"] == "1" else []
                }

        for mid in record:
            record[mid]["invite_cnt"].sort()
            record[mid]["pos_cnt"].sort()
        
        m_inv_all = []
        m_inv_before = []
        m_acpt_inv_before = []
        m_acpt_rate_before = []
        m_last_acpt_inv = []
        m_first_inv = []
        m_last_inv = []
        m_inv_time_span = []
        for data in self.invite_info("extractMemberInvite1-2/2"):
            mid = data["M_id"]
            invite_date = int(data["Invite_Time"].split("-")[0][1:])
            invite_hour = int(data["Invite_Time"].split("-")[1][1:])
            
            if mid in record:
                m_invite = [x for x in record[mid]["invite_cnt"] if x < (invite_date, invite_hour)]
                m_pos = [x for x in record[mid]["pos_cnt"] if x < (invite_date, invite_hour)]
                
                m_inv_all.append(len(record[mid]["invite_cnt"]))
                m_inv_before.append(len(m_invite))
                m_acpt_inv_before.append(len(m_pos))
                m_acpt_rate_before.append(len(m_pos)/(len(m_invite)+1))
                
                if self.train_or_test=="train":
                    m_last_acpt_inv.append((invite_date-m_pos[-1][0])*24+(invite_hour-m_pos[-1][1]) if m_pos else np.NaN)
                    first_inv = (invite_date-m_invite[0][0])*24+(invite_hour-m_invite[0][1]) if m_invite else np.NaN
                    last_inv = (invite_date-m_invite[-1][0])*24+(invite_hour-m_invite[-1][1]) if m_invite else np.NaN
                elif self.train_or_test=="test":
                    m_last_acpt_inv.append((3868-m_pos[-1][0])*24+(0-m_pos[-1][1]) if m_pos else np.NaN)
                    first_inv = (3868-m_invite[0][0])*24+(0-m_invite[0][1]) if m_invite else np.NaN
                    last_inv = (3868-m_invite[-1][0])*24+(0-m_invite[-1][1]) if m_invite else np.NaN
                elif self.train_or_test=="final":
                    m_last_acpt_inv.append((3868-m_pos[-1][0])*24+(0-m_pos[-1][1]) if m_pos else np.NaN)
                    first_inv = (3868-m_invite[0][0])*24+(0-m_invite[0][1]) if m_invite else np.NaN
                    last_inv = (3868-m_invite[-1][0])*24+(0-m_invite[-1][1]) if m_invite else np.NaN
                else:
                    raise Exception()
                
                m_first_inv.append(first_inv)
                m_last_inv.append(last_inv)
                m_inv_time_span.append(first_inv - last_inv if first_inv and last_inv else np.NaN)
                
            else:
                m_inv_all.append(0)
                m_inv_before.append(0)
                m_acpt_inv_before.append(0)
                m_acpt_rate_before.append(0)
                m_last_acpt_inv.append(np.NaN)
                m_first_inv.append(np.NaN)
                m_last_inv.append(np.NaN)
                m_inv_time_span.append(np.NaN)

        np.save(self.save_path+"m_inv_all.npy", np.array([m_inv_all]))
        np.save(self.save_path+"m_inv_before.npy", np.array([m_inv_before]))
        np.save(self.save_path+"m_acpt_inv_before.npy", np.array([m_acpt_inv_before]))
        np.save(self.save_path+"m_acpt_rate_before.npy", np.array([m_acpt_rate_before]))
        np.save(self.save_path+"m_last_acpt_inv.npy", np.array([m_last_acpt_inv]))
        np.save(self.save_path+"m_first_inv.npy", np.array([m_first_inv]))
        np.save(self.save_path+"m_last_inv.npy", np.array([m_last_inv]))
        np.save(self.save_path+"m_inv_time_span.npy", np.array([m_inv_time_span]))
    
    def extractMemberInvite2(self):
        record = {}
        for data in chain(DataReader("invite_info")("extractMemberInvite2-1/3"), 
                          DataReader("invite_info_evaluate")("extractMemberInvite2-2.5/3"),
                         DataReader("invite_info_final")("extractMemberInvite2-2.5/3")):
            mid = data["M_id"]
            invite_date = int(data["Invite_Time"].split("-")[0][1:])
            invite_hour = int(data["Invite_Time"].split("-")[1][1:])
            if mid in record:
                record[mid] += [(invite_date, invite_hour)]
            else:
                record[mid] = [(invite_date, invite_hour)]
        
        for mid in tqdm(record):
            record[mid].sort()

        m_inv_sametime_num = []
        m_inv_sameday_num = []
        m_inv_samehour_num = []
        for data in self.invite_info("extractMemberInvite2-3/3"):
            mid = data["M_id"]
            invite_date = int(data["Invite_Time"].split("-")[0][1:])
            invite_hour = int(data["Invite_Time"].split("-")[1][1:])
            
            inv_sametime_all = [x for x in record[mid] if x == (invite_date, invite_hour)]
            inv_sameday_all = [x for x in record[mid] if x[0] == invite_date]
            inv_samehour_all = [x for x in record[mid] if x[1] == invite_hour]
            
        np.save(self.save_path+"m_inv_sametime_num.npy", np.array([m_inv_sametime_num]))
        np.save(self.save_path+"m_inv_sameday_num.npy", np.array([m_inv_sameday_num]))
        np.save(self.save_path+"m_inv_samehour_num.npy", np.array([m_inv_samehour_num]))
        
    def extractQuestionBasicInfo(self):
        """
        问题的基本信息
        qtitle_sw_cnt: 问题标题的字数
        qdesc_sw_cnt:  问题内容的字数
        qtopic_cnt:    问题的标记话题数
        """
        question_table = {}
        for data in self.question_info("extractQuestionBasicInfo-1/2"):
            qid = data["Q_id"]
            qtitle_sw = data["QTitle_SW_list"]
            qdesc_sw = data["QDescrible_SW_list"]
            qtopic = data["QT_list"]
            question_table[qid] = {}
            question_table[qid]["qtitle_sw_cnt"] = len(qtitle_sw.split(",")) if qtitle_sw != "-1" else 0
            question_table[qid]["qdesc_sw_cnt"] = len(qdesc_sw.split(",")) if qdesc_sw != "-1" else 0
            question_table[qid]["qtopic_cnt"] = len(qtopic.split(",")) if qtopic != "-1" else 0
        
        qtitle_sw_cnt = []
        qdesc_sw_cnt = []
        qtopic_cnt = []
        for data in self.invite_info("extractQuestionBasicInfo-2/2"):
            qid = data["Q_id"]
            qtitle_sw_cnt.append(question_table[qid]["qtitle_sw_cnt"])
            qdesc_sw_cnt.append(question_table[qid]["qdesc_sw_cnt"])
            qtopic_cnt.append(question_table[qid]["qtopic_cnt"])
                
        np.save(self.save_path+"qtitle_sw_cnt.npy", np.array([qtitle_sw_cnt]))
        np.save(self.save_path+"qdesc_sw_cnt.npy", np.array([qdesc_sw_cnt]))
        np.save(self.save_path+"qtopic_cnt.npy", np.array([qtopic_cnt]))

    def extractMemberBasicInfo(self):
        """
        提取用户基本信息
        clf_a: 用户二分类特征a
        clf_b: 用户二分类特征b
        clf_c: 用户二分类特征c
        clf_d: 用户二分类特征d
        clf_e: 用户二分类特征e
        gender_male: 是否为男性用户
        gender_female: 是否为女性用户
        gender_unknown: 是否性别未知
        visit_freq_daily: 是否为日活用户
        visit_freq_weekly: 是否为周活用户
        visit_freq_monthly: 是否为月活用户
        visit_freq_new: 是否为新用户
        visit_freq_unknown: 是否为未知用户
        salt_value: 用户盐值
        att_topic_num: 用户关注的话题数
        int_topic_num: 用户感兴趣的话题数
        """
        member_table = {}
        for data in self.member_info("extractMemberBasicInfo-1/2"):
            mid = data["M_id"]
            member_table[mid] = data
        
        clf_a, clf_b, clf_c, clf_d, clf_e = [], [], [], [], []
        gender_male, gender_female, gender_unknown = [], [], []
        visit_freq_daily, visit_freq_weekly, visit_freq_monthly, visit_freq_new, visit_freq_unknown = [], [], [], [], []
        salt_value = []
        att_topic_num = []
        int_topic_num = []
        for data in self.invite_info("extractMemberBasicInfo-2/2"):
            mid = data["M_id"]
            
            clf_a.append(int(member_table[mid]["Feature_A"]))
            clf_b.append(int(member_table[mid]["Feature_B"]))
            clf_c.append(int(member_table[mid]["Feature_C"]))
            clf_d.append(int(member_table[mid]["Feature_D"]))
            clf_e.append(int(member_table[mid]["Feature_E"]))
            
            gender_male.append(1 if member_table[mid]["Gender"] == "male" else 0)
            gender_female.append(1 if member_table[mid]["Gender"] == "female" else 0)
            gender_unknown.append(1 if member_table[mid]["Gender"] == "unknown" else 0)
            
            visit_freq_daily.append(1 if member_table[mid]["Frequency"] == "daily" else 0)
            visit_freq_weekly.append(1 if member_table[mid]["Frequency"] == "weekly" else 0)
            visit_freq_monthly.append(1 if member_table[mid]["Frequency"] == "monthly" else 0)
            visit_freq_new.append(1 if member_table[mid]["Frequency"] == "new" else 0)
            visit_freq_unknown.append(1 if member_table[mid]["Frequency"] == "unknown" else 0)
            
            salt_value.append(int(member_table[mid]["Score"]))
            att_topic_num.append(len(member_table[mid]["T_attention"].split(",") if member_table[mid]["T_attention"] != -1 else []))
            int_topic_num.append(len(member_table[mid]["T_interest"].split(",") if member_table[mid]["T_interest"] != -1 else []))
        
        np.save(self.save_path+"clf_a.npy", np.array([clf_a]))
        np.save(self.save_path+"clf_b.npy", np.array([clf_b]))
        np.save(self.save_path+"clf_c.npy", np.array([clf_c]))
        np.save(self.save_path+"clf_d.npy", np.array([clf_d]))
        np.save(self.save_path+"clf_e.npy", np.array([clf_e]))
        
        np.save(self.save_path+"gender_male.npy", np.array([gender_male]))
        np.save(self.save_path+"gender_female.npy", np.array([gender_female]))
        np.save(self.save_path+"gender_unknown.npy", np.array([gender_unknown]))
        
        np.save(self.save_path+"visit_freq_daily.npy", np.array([visit_freq_daily]))
        np.save(self.save_path+"visit_freq_weekly.npy", np.array([visit_freq_weekly]))
        np.save(self.save_path+"visit_freq_monthly.npy", np.array([visit_freq_monthly]))
        np.save(self.save_path+"visit_freq_new.npy", np.array([visit_freq_new]))
        np.save(self.save_path+"visit_freq_unknown.npy", np.array([visit_freq_unknown]))
        
        np.save(self.save_path+"salt_value.npy", np.array([salt_value]))
        np.save(self.save_path+"att_topic_num.npy", np.array([att_topic_num]))
        np.save(self.save_path+"int_topic_num.npy", np.array([int_topic_num]))
   

    def extractQuestionHistoryAnswerSta(self):
        """
        问题的历史回答统计特征
        q_ans_num: 问题的历史回答数量
        q_ans_like_sum: 问题的历史回答点赞数总和
        q_ans_cmt_sum: 问题的历史回答评论数总和
        q_ans_pic_sum: 问题的历史回答存在图片数总和
        q_ans_thx_sum: 问题的历史回答感谢数总和
        q_ans_clc_sum: 问题的历史回答收藏数总和
        q_ans_sw_sum: 问题的历史回答字数总和
        q_ans_sw_mean: 问题的历史回答字数平均值
        q_last_ans: 问题的上一个问答距今多久
        q_first_ans: 问题的第一个回答距今多久
        q_ans_time_span: 问题的历史回答的时间跨度
        q_ans_freq_mean: 问题的历史回答的回答频率均值
        q_ans_freq_std: 问题的历史回答的回答频率标准差
        """
        qid2answer = {}
        for data in self.answer_info("extractQuestionHistoryAnswerSta-1/2"):
            qid = data["Q_id"]
            date = int(data["A_create_time"].split("-")[0][1:])
            hour = int(data["A_create_time"].split("-")[1][1:])
            data["time"] = (date, hour)
            if qid in qid2answer:
                qid2answer[qid].append(data)
            else:
                qid2answer[qid] = [data]
        
        q_ans_num = []
        q_ans_like_sum = []
        q_ans_cmt_sum = []
        q_ans_pic_sum = []
        q_ans_thx_sum = []
        q_ans_clc_sum = []
        q_ans_sw_sum = []
        q_ans_sw_mean = []
        q_last_ans = []
        q_first_ans = []
        q_ans_time_span = []
        q_ans_freq_mean = []
        q_ans_freq_std = []
        for data in self.invite_info("extractQuestionHistoryAnswerSta-2/2"):
            qid = data["Q_id"]
            invite_date = int(data["Invite_Time"].split("-")[0][1:])
            invite_hour = int(data["Invite_Time"].split("-")[1][1:])
            
            if self.train_or_test=="train":
                ans = [x for x in qid2answer[qid] if (invite_date-30,0)<x["time"]<(invite_date, invite_hour)] if qid in qid2answer else []
            elif self.train_or_test=="test":
                ans = [x for x in qid2answer[qid] if (3868-30,0)<x["time"]<(invite_date, invite_hour)] if qid in qid2answer else []
            else:
                raise Exception()
            
            ans_time = sorted([x["time"] for x in ans])

            ans_num = len(ans)
            ans_like_sum = sum([int(x["Count_Like"]) for x in ans])
            ans_cmt_sum = sum([int(x["Count_comment"]) for x in ans])
            ans_pic_sum = sum([int(x["IncludePicture"]) for x in ans])
            ans_thx_sum = sum([int(x["Count_Thank"]) for x in ans])
            ans_clc_sum = sum([int(x["Count_Collect"]) for x in ans])
            ans_sw_sum = sum([int(x["Count_SW"]) for x in ans])
            ans_sw_mean = ans_sw_sum/ans_num if ans_num else 0
            
            if self.train_or_test=="train":
                last_ans = (invite_date-ans_time[-1][0])*24+(invite_hour-ans_time[-1][1]) if ans_time else np.NaN
                first_ans = (invite_date-ans_time[0][0])*24+(invite_hour-ans_time[0][1]) if ans_time else np.NaN
            elif self.train_or_test=="test":
                last_ans = (3868-ans_time[-1][0])*24+(0-ans_time[-1][1]) if ans_time else np.NaN
                first_ans = (3868-ans_time[0][0])*24+(0-ans_time[0][1]) if ans_time else np.NaN
            else:
                raise Exception()
            
            ans_time_span = first_ans - last_ans if first_ans and last_ans else np.NaN
            ans_freq = [(b[0]-a[0])*24+(b[1]-a[1]) for a,b in zip(ans_time, ans_time[1:])] if ans_time else []
            ans_freq_mean = np.mean(ans_freq) if ans_freq else np.NaN
            ans_freq_std = np.std(ans_freq) if ans_freq else np.NaN
            
            q_ans_num.append(ans_num)
            q_ans_like_sum.append(ans_like_sum) 
            q_ans_cmt_sum.append(ans_cmt_sum) 
            q_ans_pic_sum.append(ans_pic_sum) 
            q_ans_thx_sum.append(ans_thx_sum) 
            q_ans_clc_sum.append(ans_clc_sum) 
            q_ans_sw_sum.append(ans_sw_sum) 
            q_ans_sw_mean.append(ans_sw_mean)
            q_last_ans.append(last_ans)
            q_first_ans.append(first_ans)
            q_ans_time_span.append(ans_time_span)
            q_ans_freq_mean.append(ans_freq_mean)
            q_ans_freq_std.append(ans_freq_std)
        
        np.save(self.save_path+"q_ans_num.npy", np.array([q_ans_num]))
        np.save(self.save_path+"q_ans_like_sum.npy", np.array([q_ans_like_sum]))
        np.save(self.save_path+"q_ans_cmt_sum.npy", np.array([q_ans_cmt_sum]))
        np.save(self.save_path+"q_ans_pic_sum.npy", np.array([q_ans_pic_sum]))
        np.save(self.save_path+"q_ans_thx_sum.npy", np.array([q_ans_thx_sum]))
        np.save(self.save_path+"q_ans_clc_sum.npy", np.array([q_ans_clc_sum]))
        np.save(self.save_path+"q_ans_sw_sum.npy", np.array([q_ans_sw_sum]))
        np.save(self.save_path+"q_ans_sw_mean.npy", np.array([q_ans_sw_mean]))
        np.save(self.save_path+"q_last_ans.npy", np.array([q_last_ans]))
        np.save(self.save_path+"q_first_ans.npy", np.array([q_first_ans]))
        np.save(self.save_path+"q_ans_time_span.npy", np.array([q_ans_time_span]))
        np.save(self.save_path+"q_ans_freq_mean.npy", np.array([q_ans_freq_mean]))
        np.save(self.save_path+"q_ans_freq_std.npy", np.array([q_ans_freq_std]))
    
    def extractMemberHistoryAnswerSta(self):
        """
        用户的历史回答统计特征
        m_ans_num: 用户历史回答的总数
        m_ans_like_sum: 用户历史回答的点赞数总和
        m_ans_cmt_sum: 用户历史回答的评论数总和
        m_ans_pic_sum: 用户历史回答的存在图片数总和
        m_ans_thx_sum: 用户历史回答的感谢数总和
        m_ans_clc_sum: 用户历史回答的收藏数总和
        m_ans_sw_sum: 用户历史回答的字数总和
        m_ans_sw_mean: 用户历史回答的字数均值
        m_last_ans: 用户历史回答的上一次回答距今多久
        m_first_ans: 用户历史回答的第一次回答距今多久
        m_ans_time_span: 用户历史回答的时间跨度
        m_ans_freq_mean: 用户历史回答的回答频率均值
        m_ans_freq_std: 用户历史回答的回答频率标准差
        """
        mid2answer = {}
        for data in self.answer_info("extractMemberHistoryAnswerSta-1/2"):
            mid = data["M_id"]
            date = int(data["A_create_time"].split("-")[0][1:])
            hour = int(data["A_create_time"].split("-")[1][1:])
            data["time"] = (date, hour)
            if mid in mid2answer:
                mid2answer[mid].append(data)
            else:
                mid2answer[mid] = [data]
        
        m_ans_num = []
        m_ans_like_sum = []
        m_ans_cmt_sum = []
        m_ans_pic_sum = []
        m_ans_thx_sum = []
        m_ans_clc_sum = []
        m_ans_sw_sum = []
        m_ans_sw_mean = []
        m_last_ans = []
        m_first_ans = []
        m_ans_time_span = []
        m_ans_freq_mean = []
        m_ans_freq_std = []
        for data in self.invite_info("extractMemberHistoryAnswerSta-2/2"):
            mid = data["M_id"]
            invite_date = int(data["Invite_Time"].split("-")[0][1:])
            invite_hour = int(data["Invite_Time"].split("-")[1][1:])
            
            if self.train_or_test=="train":
                ans = [x for x in mid2answer[mid] if (invite_date-30,0)<x["time"]<(invite_date, invite_hour)] if mid in mid2answer else []
            elif self.train_or_test=="test":
                ans = [x for x in mid2answer[mid] if (3868-30,0)<x["time"]<(invite_date, invite_hour)] if mid in mid2answer else []
            elif self.train_or_test=="final":
                ans = [x for x in mid2answer[mid] if (3868-30,0)<x["time"]<(invite_date, invite_hour)] if mid in mid2answer else []
            else:
                raise Exception()
                
            ans_time = sorted([x["time"] for x in ans])

            ans_num = len(ans)
            ans_like_sum = sum([int(x["Count_Like"]) for x in ans])
            ans_cmt_sum = sum([int(x["Count_comment"]) for x in ans])
            ans_pic_sum = sum([int(x["IncludePicture"]) for x in ans])
            ans_thx_sum = sum([int(x["Count_Thank"]) for x in ans])
            ans_clc_sum = sum([int(x["Count_Collect"]) for x in ans])
            ans_sw_sum = sum([int(x["Count_SW"]) for x in ans])
            ans_sw_mean = ans_sw_sum/ans_num if ans_num else 0
            
            if self.train_or_test=="train":
                last_ans = (invite_date-ans_time[-1][0])*24+(invite_hour-ans_time[-1][1]) if ans_time else np.NaN
                first_ans = (invite_date-ans_time[0][0])*24+(invite_hour-ans_time[0][1]) if ans_time else np.NaN
            elif self.train_or_test=="test":
                last_ans = (3868-ans_time[-1][0])*24+(0-ans_time[-1][1]) if ans_time else np.NaN
                first_ans = (3868-ans_time[0][0])*24+(0-ans_time[0][1]) if ans_time else np.NaN
            elif self.train_or_test=="final":
                last_ans = (3868-ans_time[-1][0])*24+(0-ans_time[-1][1]) if ans_time else np.NaN
                first_ans = (3868-ans_time[0][0])*24+(0-ans_time[0][1]) if ans_time else np.NaN
            else:
                raise Exception()
            
            ans_time_span = first_ans - last_ans if first_ans and last_ans else np.NaN
            ans_freq = [(b[0]-a[0])*24+(b[1]-a[1]) for a,b in zip(ans_time, ans_time[1:])] if ans_time else []
            ans_freq_mean = np.mean(ans_freq) if ans_freq else np.NaN
            ans_freq_std = np.std(ans_freq) if ans_freq else np.NaN
            
            m_ans_num.append(ans_num)
            m_ans_like_sum.append(ans_like_sum) 
            m_ans_cmt_sum.append(ans_cmt_sum) 
            m_ans_pic_sum.append(ans_pic_sum) 
            m_ans_thx_sum.append(ans_thx_sum) 
            m_ans_clc_sum.append(ans_clc_sum) 
            m_ans_sw_sum.append(ans_sw_sum) 
            m_ans_sw_mean.append(ans_sw_mean)
            m_last_ans.append(last_ans)
            m_first_ans.append(first_ans)
            m_ans_time_span.append(ans_time_span)
            m_ans_freq_mean.append(ans_freq_mean)
            m_ans_freq_std.append(ans_freq_std)
        
        np.save(self.save_path+"m_ans_num.npy", np.array([m_ans_num]))
        np.save(self.save_path+"m_ans_like_sum.npy", np.array([m_ans_like_sum]))
        np.save(self.save_path+"m_ans_cmt_sum.npy", np.array([m_ans_cmt_sum]))
        np.save(self.save_path+"m_ans_pic_sum.npy", np.array([m_ans_pic_sum]))
        np.save(self.save_path+"m_ans_thx_sum.npy", np.array([m_ans_thx_sum]))
        np.save(self.save_path+"m_ans_clc_sum.npy", np.array([m_ans_clc_sum]))
        np.save(self.save_path+"m_ans_sw_sum.npy", np.array([m_ans_sw_sum]))
        np.save(self.save_path+"m_ans_sw_mean.npy", np.array([m_ans_sw_mean]))
        np.save(self.save_path+"m_last_ans.npy", np.array([m_last_ans]))
        np.save(self.save_path+"m_first_ans.npy", np.array([m_first_ans]))
        np.save(self.save_path+"m_ans_time_span.npy", np.array([m_ans_time_span]))
        np.save(self.save_path+"m_ans_freq_mean.npy", np.array([m_ans_freq_mean]))
        np.save(self.save_path+"m_ans_freq_std.npy", np.array([m_ans_freq_std]))

    def extractMemberAcptInv(self):
        """
        用户接受邀请次数（K-Folder）
        m_acpt_inv_all: 用户接受邀请次数
        m_acpt_rate_all: 用户接受邀请率
        """
        record = {}
        for data in DataReader("invite_info")("extractMemberAcptInv-1/3"):
            mid = data["M_id"]
            label = data["Label"]
            folder = data["Folder"]
            
            if mid in record:
                record[mid]["cnt"][folder] += 1 
                record[mid]["pos"][folder] += 1 if label == "1" else 0
            else:
                record[mid] = {"cnt":{"0":0, "1":0, "2":0, "3":0, "4":0, "5":0},"pos":{"0":0, "1":0, "2":0, "3":0, "4":0, "5":0}}
                record[mid]["cnt"][folder] += 1 
                record[mid]["pos"][folder] += 1 if label == "1" else 0
        
        m_acpt_inv_all = []
        m_acpt_rate_all = []
        for data in self.invite_info("extractMemberAcptInv-3/3"):
            mid = data["M_id"]
            folder = data["Folder"] if "Folder" in data else 10
            
            if mid in record:
                acpt_inv = sum([record[mid]["pos"][str(i)] for i in range(0,6) if str(i) != folder])
                inv_cnt = sum([record[mid]["cnt"][str(i)] for i in range(0,6) if str(i) != folder])
                acpt_rate = acpt_inv/inv_cnt if inv_cnt!=0 else 0
            else:
                acpt_inv = 0
                acpt_rate = 0
            
            m_acpt_inv_all.append(acpt_inv)
            m_acpt_rate_all.append(acpt_rate)
            
        np.save(self.save_path+"m_acpt_inv_all.npy", np.array([m_acpt_inv_all]))
        np.save(self.save_path+"m_acpt_rate_all.npy", np.array([m_acpt_rate_all]))

    def extractQuestionAcptInv(self):
        """
        问题接受邀请次数（K-Folder）
        q_acpt_inv_all: 问题接受邀请次数
        q_acpt_rate_all: 问题接受邀请率
        """
        record = {}
        for data in DataReader("invite_info")("extractQuestionAcptInv-1/2"):
            qid = data["Q_id"]
            label = data["Label"]
            folder = data["Folder"]
            
            if qid in record:
                record[qid]["cnt"][folder] += 1 
                record[qid]["pos"][folder] += 1 if label == "1" else 0
            else:
                record[qid] = {"cnt":{"0":0, "1":0, "2":0, "3":0, "4":0, "5":0},"pos":{"0":0, "1":0, "2":0, "3":0, "4":0, "5":0}}
                record[qid]["cnt"][folder] += 1 
                record[qid]["pos"][folder] += 1 if label == "1" else 0
                
        q_acpt_inv_all = []
        q_acpt_rate_all = []
        for data in self.invite_info("extractQuestionAcptInv-2/2"):
            qid = data["Q_id"]
            folder = data["Folder"] if "Folder" in data else 10
            if qid in record:
                acpt_inv = sum([record[qid]["pos"][str(i)] for i in range(0,6) if str(i) != folder])
                inv_cnt = sum([record[qid]["cnt"][str(i)] for i in range(0,6) if str(i) != folder])
                acpt_rate = acpt_inv/inv_cnt if inv_cnt!=0 else 0
            else:
                acpt_inv = 0
                acpt_rate = 0
        
            q_acpt_inv_all.append(acpt_inv)
            q_acpt_rate_all.append(acpt_rate)
            
        np.save(self.save_path+"q_acpt_inv_all.npy", np.array([q_acpt_inv_all]))
        np.save(self.save_path+"q_acpt_rate_all.npy", np.array([q_acpt_rate_all]))
        
    
    def extractTopicDist(self):
        """
        用户到问题关于话题的距离
        """
        def euc_dist(t1, t2):
            return np.sqrt(np.sum(np.square(tid2vector[t1]-tid2vector[t2])))
        
        def cos_dist(t1, t2):
            return np.dot(tid2vector[t1],tid2vector[t2])/(np.linalg.norm(tid2vector[t1])*(np.linalg.norm(tid2vector[t2])))
        
        tid2vector = {}
        for data in self.topic_vectors("extractTopicSim-1/4"):
            tid2vector[data["T_id"]] = np.array([float(x) for x in data["T_embeding"].split(" ")])
            
        qid2tid = {}
        cnt = 0
        for data in self.question_info("extractTopicSim-2/4"):
            qid2tid[data["Q_id"]] = data["QT_list"].split(",") if data["QT_list"] != "-1" else []
        
        mid2tid = {}
        for data in self.member_info("extractTopicSim-3/4"):
            mid = data["M_id"]
            mid2tid[mid] = {}
            mid2tid[mid]["attention"] = data['T_attention'].split(",") if data['T_attention'] != "-1" else []
            mid2tid[mid]["interest"] = {}
            for x in data['T_interest'].split(","):
                if x != "-1":
                    tid_weight = x.split(":")
                    mid2tid[mid]["interest"][tid_weight[0]] = float(tid_weight[1])
        
        m_att_euc_dist = []
        q_att_euc_dist = []
        m_int_euc_dist = []
        q_int_euc_dist = []
        m_att_cos_dist = []
        q_att_cos_dist = []
        m_int_cos_dist = []
        q_int_cos_dist = []
        for data in self.invite_info("extractTopicSim-4/4"):
            q_tid = qid2tid[data["Q_id"]]
            m_att_tid = mid2tid[data["M_id"]]["attention"]
            m_int_tid = mid2tid[data["M_id"]]["interest"]
            
            m_att_euc = np.mean([min([euc_dist(qt, mt) for qt in q_tid]) for mt in m_att_tid]) if q_tid and m_att_tid else np.NaN
            q_att_euc = np.mean([min([euc_dist(qt, mt) for mt in m_att_tid]) for qt in q_tid]) if q_tid and m_att_tid else np.NaN
            m_int_euc = np.mean([min([euc_dist(qt, mt) for qt in q_tid])/m_int_tid[mt] for mt in m_int_tid]) if q_tid and m_int_tid else np.NaN
            q_int_euc = np.mean([min([euc_dist(qt, mt)/m_int_tid[mt] for mt in m_int_tid]) for qt in q_tid]) if q_tid and m_int_tid else np.NaN
            
            m_att_euc_dist.append(m_att_euc)
            q_att_euc_dist.append(q_att_euc)
            m_int_euc_dist.append(m_int_euc)
            q_int_euc_dist.append(q_int_euc)
            
        np.save(self.save_path+"m_att_dist.npy", np.array([m_att_euc_dist]))
        np.save(self.save_path+"q_att_dist.npy", np.array([q_att_euc_dist]))
        np.save(self.save_path+"m_int_dist.npy", np.array([m_int_euc_dist]))
        np.save(self.save_path+"q_int_dist.npy", np.array([q_int_euc_dist]))

    def extractNodeDensity(self):
        """
        提取问题和用户的周围密度
        """
        qid2tid = {}
        tid2qid = {}
        for data in self.question_info("extractNodeDensity-1"):
            qid = data["Q_id"]
            tids = data["QT_list"].split(",") if data["QT_list"] != "-1" else []
            qid2tid[qid] = tids
            for tid in tids:
                if tid in tid2qid:
                    tid2qid[tid].append(qid)
                else:
                    tid2qid[tid] = [qid]
        
        mid_int_weight = {}
        mid2att_tid = {}
        mid2int_tid = {}
        att_tid2mid = {}
        int_tid2mid = {}
        for data in self.member_info("extractNodeDensity-2"):
            mid = data["M_id"]
            att_tids = data["T_attention"].split(",") if data["T_attention"] != "-1" else []
            int_tids_weight = {}
            for x in data['T_interest'].split(","):
                if x != "-1":
                    tid_weight = x.split(":")
                    int_tids_weight[tid_weight[0]] = float(tid_weight[1]) if tid_weight[1] != "Infinity" else 4
            int_tids = int_tids_weight.keys()
            
            mid_int_weight[mid] = int_tids_weight
            mid2att_tid[mid] = att_tids
            mid2int_tid[mid] = int_tids
            
            for tid in att_tids:
                if tid in att_tid2mid:
                    att_tid2mid[tid].append(mid)
                else:
                    att_tid2mid[tid] = [mid]
            
            for tid in int_tids:
                if tid in int_tid2mid:
                    int_tid2mid[tid].append(mid)
                else:
                    int_tid2mid[tid] = [mid]
                    
        q_q_density = {}
        q_m_att_density = {}
        q_m_int_density = {}
        for data in self.question_info("extractNodeDensity-3"):
            qid = data["Q_id"]
            q_q_density[qid] = [len(tid2qid[tid]) for tid in qid2tid[qid]]
            q_m_att_density[qid] = [len(att_tid2mid[tid]) if tid in att_tid2mid else 0 for tid in qid2tid[qid]]
            q_m_int_density[qid] = [len(int_tid2mid[tid]) if tid in int_tid2mid else 0 for tid in qid2tid[qid]]
        
        m_m_att_density = {}
        m_m_int_density = {}
        m_q_att_density = {}
        m_q_int_density = {}
        for data in self.member_info("extractNodeDensity-4"):
            mid = data["M_id"]
            m_m_att_density[mid] = [len(att_tid2mid[tid]) for tid in mid2att_tid[mid]]
            m_m_int_density[mid] = [len(int_tid2mid[tid]) for tid in mid2int_tid[mid]]
            m_q_att_density[mid] = [len(tid2qid[tid]) if tid in tid2qid else 0 for tid in mid2att_tid[mid]]
            m_q_int_density[mid] = [len(tid2qid[tid]) if tid in tid2qid else 0 for tid in mid2int_tid[mid]]
    
        q_q_density_sum = []
        q_q_density_mean = []
        q_q_density_median = []
        q_q_density_max = []
        q_q_density_min = []
        q_m_att_density_sum = []
        q_m_att_density_mean = []
        q_m_att_density_median = []
        q_m_att_density_max = []
        q_m_att_density_min = []
        q_m_int_density_sum = []
        q_m_int_density_mean = []
        q_m_int_density_median = []
        q_m_int_density_max = []
        q_m_int_density_min = []
        m_m_att_density_sum = []
        m_m_att_density_mean = []
        m_m_att_density_median = []
        m_m_att_density_max = []
        m_m_att_density_min = []
        m_m_int_density_sum = []
        m_m_int_density_mean = []
        m_m_int_density_median = []
        m_m_int_density_max = []
        m_m_int_density_min = []
        m_q_att_density_sum = []
        m_q_att_density_mean = []
        m_q_att_density_median = []
        m_q_att_density_max = []
        m_q_att_density_min = []
        m_q_int_density_sum = []
        m_q_int_density_mean = []
        m_q_int_density_median = []
        m_q_int_density_max = []
        m_q_int_density_min = []
        for data in self.invite_info("extractNodeDensity-5"):
            qid = data["Q_id"]
            mid = data["M_id"]
            
            q_q_density_sum.append(sum(q_q_density[qid]))
            q_q_density_mean.append(np.mean(q_q_density[qid]) if q_q_density[qid] else np.NaN)
            q_q_density_median.append(np.median(q_q_density[qid]) if q_q_density[qid] else np.NaN)
            q_q_density_max.append(max(q_q_density[qid]) if q_q_density[qid] else np.NaN)
            q_q_density_min.append(min(q_q_density[qid]) if q_q_density[qid] else np.NaN)
            
            q_m_att_density_sum.append(sum(q_m_att_density[qid]))
            q_m_att_density_mean.append(np.mean(q_m_att_density[qid]) if q_m_att_density[qid] else np.NaN)
            q_m_att_density_median.append(np.median(q_m_att_density[qid]) if q_m_att_density[qid] else np.NaN)
            q_m_att_density_max.append(max(q_m_att_density[qid]) if q_m_att_density[qid] else np.NaN)
            q_m_att_density_min.append(min(q_m_att_density[qid]) if q_m_att_density[qid] else np.NaN)
            
            q_m_int_density_sum.append(sum(q_m_int_density[qid]))
            q_m_int_density_mean.append(np.mean(q_m_int_density[qid]) if q_m_int_density[qid] else np.NaN)
            q_m_int_density_median.append(np.median(q_m_int_density[qid]) if q_m_int_density[qid] else np.NaN)
            q_m_int_density_max.append(max(q_m_int_density[qid]) if q_m_int_density[qid] else np.NaN)
            q_m_int_density_min.append(min(q_m_int_density[qid]) if q_m_int_density[qid] else np.NaN)
            
            m_m_att_density_sum.append(sum(m_m_att_density[mid]))
            m_m_att_density_mean.append(np.mean(m_m_att_density[mid]) if m_m_att_density[mid] else np.NaN)
            m_m_att_density_median.append(np.median(m_m_att_density[mid]) if m_m_att_density[mid] else np.NaN)
            m_m_att_density_max.append(max(m_m_att_density[mid]) if m_m_att_density[mid] else np.NaN)
            m_m_att_density_min.append(min(m_m_att_density[mid]) if m_m_att_density[mid] else np.NaN)
            
            m_m_int_density_sum.append(sum(m_m_int_density[mid]))
            m_m_int_density_mean.append(np.mean(m_m_int_density[mid]) if m_m_int_density[mid] else np.NaN)
            m_m_int_density_median.append(np.median(m_m_int_density[mid]) if m_m_int_density[mid] else np.NaN)
            m_m_int_density_max.append(max(m_m_int_density[mid]) if m_m_int_density[mid] else np.NaN)
            m_m_int_density_min.append(min(m_m_int_density[mid]) if m_m_int_density[mid] else np.NaN)
            
            m_q_att_density_sum.append(sum(m_q_att_density[mid]))
            m_q_att_density_mean.append(np.mean(m_q_att_density[mid]) if m_q_att_density[mid] else np.NaN)
            m_q_att_density_median.append(np.median(m_q_att_density[mid]) if m_q_att_density[mid] else np.NaN)
            m_q_att_density_max.append(max(m_q_att_density[mid]) if m_q_att_density[mid] else np.NaN)
            m_q_att_density_min.append(min(m_q_att_density[mid]) if m_q_att_density[mid] else np.NaN)
            
            m_q_int_density_sum.append(sum(m_q_int_density[mid]))
            m_q_int_density_mean.append(np.mean(m_q_int_density[mid]) if m_q_int_density[mid] else np.NaN)
            m_q_int_density_median.append(np.median(m_q_int_density[mid]) if m_q_int_density[mid] else np.NaN)
            m_q_int_density_max.append(max(m_q_int_density[mid]) if m_q_int_density[mid] else np.NaN)
            m_q_int_density_min.append(min(m_q_int_density[mid]) if m_q_int_density[mid] else np.NaN)
            
        np.save(self.save_path+"q_q_density_sum.npy", np.array([q_q_density_sum]))
        np.save(self.save_path+"q_q_density_mean.npy", np.array([q_q_density_mean]))
        np.save(self.save_path+"q_q_density_median.npy", np.array([q_q_density_median]))
        np.save(self.save_path+"q_q_density_max.npy", np.array([q_q_density_max]))
        np.save(self.save_path+"q_q_density_min.npy", np.array([q_q_density_min]))
        np.save(self.save_path+"q_m_att_density_sum.npy", np.array([q_m_att_density_sum]))
        np.save(self.save_path+"q_m_att_density_mean.npy", np.array([q_m_att_density_mean]))
        np.save(self.save_path+"q_m_att_density_median.npy", np.array([q_m_att_density_median]))
        np.save(self.save_path+"q_m_att_density_max.npy", np.array([q_m_att_density_max]))
        np.save(self.save_path+"q_m_att_density_min.npy", np.array([q_m_att_density_min]))
        np.save(self.save_path+"q_m_int_density_sum.npy", np.array([q_m_int_density_sum]))
        np.save(self.save_path+"q_m_int_density_mean.npy", np.array([q_m_int_density_mean]))
        np.save(self.save_path+"q_m_int_density_median.npy", np.array([q_m_int_density_median]))
        np.save(self.save_path+"q_m_int_density_max.npy", np.array([q_m_int_density_max]))
        np.save(self.save_path+"q_m_int_density_min.npy", np.array([q_m_int_density_min]))
        np.save(self.save_path+"m_m_att_density_sum.npy", np.array([m_m_att_density_sum]))
        np.save(self.save_path+"m_m_att_density_mean.npy", np.array([m_m_att_density_mean]))
        np.save(self.save_path+"m_m_att_density_median.npy", np.array([m_m_att_density_median]))
        np.save(self.save_path+"m_m_att_density_max.npy", np.array([m_m_att_density_max]))
        np.save(self.save_path+"m_m_att_density_min.npy", np.array([m_m_att_density_min]))
        np.save(self.save_path+"m_m_int_density_sum.npy", np.array([m_m_int_density_sum]))
        np.save(self.save_path+"m_m_int_density_mean.npy", np.array([m_m_int_density_mean]))
        np.save(self.save_path+"m_m_int_density_median.npy", np.array([m_m_int_density_median]))
        np.save(self.save_path+"m_m_int_density_max.npy", np.array([m_m_int_density_max]))
        np.save(self.save_path+"m_m_int_density_min.npy", np.array([m_m_int_density_min]))
        np.save(self.save_path+"m_q_att_density_sum.npy", np.array([m_q_att_density_sum]))
        np.save(self.save_path+"m_q_att_density_mean.npy", np.array([m_q_att_density_mean]))
        np.save(self.save_path+"m_q_att_density_median.npy", np.array([m_q_att_density_median]))
        np.save(self.save_path+"m_q_att_density_max.npy", np.array([m_q_att_density_max]))
        np.save(self.save_path+"m_q_att_density_min.npy", np.array([m_q_att_density_min]))
        np.save(self.save_path+"m_q_int_density_sum.npy", np.array([m_q_int_density_sum]))
        np.save(self.save_path+"m_q_int_density_mean.npy", np.array([m_q_int_density_mean]))
        np.save(self.save_path+"m_q_int_density_median.npy", np.array([m_q_int_density_median]))
        np.save(self.save_path+"m_q_int_density_max.npy", np.array([m_q_int_density_max]))
        np.save(self.save_path+"m_q_int_density_min.npy", np.array([m_q_int_density_min]))
```


```python
"""
抽取训练集/验证集/测试集的特征
"""

extractor = FeatureExtractor("final")

# 提取问题基本信息
extractor.extractQuestionBasicInfo()

# 提取用户的基本信息
extractor.extractMemberBasicInfo()

# 提取问题受邀时间特征
extractor.extractInvitedTime()


# 提取问题受邀特征
extractor.extractQuestionInvite()

# 提取用户的受邀特征
extractor.extractMemberInvite1()
extractor.extractMemberInvite2()

# 提取用户的历史回答统计特征
extractor.extractMemberHistoryAnswerSta()


# 提取用户的接受邀请次数
extractor.extractMemberAcptInv()


# 提取问题和用户的周围密度
extractor.extractNodeDensity()

# # topic距离
extractor.extractTopicDist()
```

### 图特征
    GraphBuilder用于构建图模型，图中的节点包括用户节点（mid），问题节点（qid），话题节点（tid）；图中的边包括用户节点-问题节点（代表用户回答问题的关系），用户节点-话题节点（代表用户关注或感兴趣的话题的关系），问题节点-话题节点（代表问题的标注话题的关系）。
    图特征有两类：
    1.问题关联度特征（extractQ2QCorrelation）：该特征表征图模型中用户回答过的问题与受邀问题的关联度，其中关联度为两者的一阶邻点集的交集的长度。
    2.路径模式特征(extractPathPattern)：该特征表征图模型中受邀用户与受邀问题之间的各模式的路径的数量，该特征涉及的路径模式包括：mid-qid-tid-qid， mid-tid-mid-qid, mid-qid-mid-qid, mid-tid-qid。


```python
import networkx as nx
from networkx import NetworkXError

class GraphBuilder(object):
    """
    图的构建器
    """
    def __init__(self):
        self.graph = nx.Graph()
        
        self.__extractMember()
        self.__extractQuestion()
        self.__extractAnswer()
    
    def __extractMember(self):
        """
        加入mid节点和tid节点，以及<mid, tid>边
        """
        for data in DataReader("member_info")("__extractMember"):
            mid = data["M_id"]
            self.graph.add_node(mid)
            
            # 加入感兴趣的话题（带权重）
            for tid_weight in data["T_interest"].split(","):
                if tid_weight == "-1":
                    continue
                tid_weight = tid_weight.split(":")
                tid = tid_weight[0] 
                weight = float(tid_weight[1]) if tid_weight[1] != "Infinity" else 5
                self.graph.add_node(tid)
                self.graph.add_edge(mid, tid)
                self.graph[mid][tid]["weight"] = weight
                
#             加入关注的话题（不带权重）
            for tid in data["T_attention"].split(","):
                if tid == "-1":
                    continue
                self.graph.add_node(tid)
                self.graph.add_edge(mid, tid)
    
    def __extractQuestion(self):
        """
        加入qid节点和tid节点，以及<qid, tid>边
        """
        for data in DataReader("question_info")("__extractQuestion"):
            qid = data["Q_id"]
            self.graph.add_node(qid)
            tids = data["QT_list"].split(",")
            for tid in tids:
                if tid == "-1":
                    continue
                self.graph.add_node(tid)
                self.graph.add_edge(qid, tid)
    
    def __extractAnswer(self):
        """
        加入<mid, qid>边
        """
        for data in DataReader("answer_info")("__extractAnswer"):
            qid = data["Q_id"]
            mid = data["M_id"]
            answer_date = int(data["A_create_time"].split("-")[0][1:])
            answer_hour = int(data["A_create_time"].split("-")[1][1:])
            self.graph.add_edge(mid, qid)
            self.graph[mid][qid]["time"] = (answer_date, answer_hour)
            
def extractQ2QCorrelation(train_or_test):
    """
    提取用户回答过的问题到目标问题的关联度
    """
    if train_or_test not in ["train", "test", "final"]:
        raise Exception("[train_or_test value is wrong]")
   
    q2q_corr_sum = []
    q2q_corr_len = []
    q2q_corr_mean = []
    q2q_corr_median = []
    q2q_corr_max = []
    q2q_corr_min = []
    
    reader = DataReader("invite_info") if train_or_test == "train" else DataReader("invite_info_final")
    for data in reader("extractQ2QCorrelation"):
        qid = data["Q_id"]
        mid = data["M_id"]
        inv_date = int(data["Invite_Time"].split("-")[0][1:])
        inv_hour = int(data["Invite_Time"].split("-")[1][1:])
        
        qid_t_m = set([x for x in gb.graph[qid] if x[0]=="T" or gb.graph[qid][x]["time"] < (inv_date, inv_hour)])
        
        corr = []
        for q in filter(lambda x: x[0]=="Q" and gb.graph[mid][x]["time"]<(inv_date, inv_hour), gb.graph[mid]):
            q2_t_m = set([x for x in gb.graph[q] if x[0]=="T" or gb.graph[q][x]["time"] < (inv_date, inv_hour)])
            corr.append(len(qid_t_m & q2_t_m))
            
        if corr:
            q2q_corr_sum.append(sum(corr))
            q2q_corr_len.append(len(corr))
            q2q_corr_mean.append(np.mean(corr))
        else:
            q2q_corr_sum.append(0)
            q2q_corr_len.append(0)
            q2q_corr_mean.append(0)
    
    np.save("features/" + train_or_test + "/" + "q2q_corr_sum.npy", np.array([q2q_corr_sum]))
    np.save("features/" + train_or_test + "/" + "q2q_corr_len.npy", np.array([q2q_corr_len]))
    np.save("features/" + train_or_test + "/" + "q2q_corr_mean.npy", np.array([q2q_corr_mean]))

def extractPathPattern(train_or_test):
    """
    提取用户到问题的路径模式
    """
    if train_or_test not in ["train", "test", "final"]:
        raise Exception("[train_or_test value is wrong]")
    
    mqtq_path_cnt = []
    mtmq_path_cnt = []
    mqmq_path_cnt = []
    mtq_path_cnt = []
    reader = DataReader("invite_info") if train_or_test == "train" else DataReader("invite_info_final")
    for data in reader("extractPathPattern"):
        qid = data["Q_id"]
        mid = data["M_id"]
        inv_date = int(data["Invite_Time"].split("-")[0][1:])
        inv_hour = int(data["Invite_Time"].split("-")[1][1:])
        
        m2t = set([x for x in gb.graph[mid] if x[0]=="T"])
        m2q = set([x for x in gb.graph[mid] if x[0]=="Q" and gb.graph[mid][x]["time"]<(inv_date, inv_hour)])
        q2t = set([x for x in gb.graph[qid] if x[0]=="T"])
        q2m = set([x for x in gb.graph[qid] if x[0]=="M" and gb.graph[qid][x]["time"]<(inv_date, inv_hour)])
    
        mqtq = 0
        for q in m2q:
            q_t = set([x for x in gb.graph[q] if x[0]=="T"])
            mqtq += len(q_t & q2t)
        
        mtmq = 0
        for m in q2m:
            m_t = set([x for x in gb.graph[m] if x[0]=="T"])
            mtmq += len(m_t & m2t)
            
        mqmq = 0
        for q in m2q:
            q_m = set([x for x in gb.graph[q] if x[0]=="M" and x != mid])
            mqmq += len(q_m & q2m)
            
        mtq = len(m2t & q2t)
        
        mqtq_path_cnt.append(mqtq)
        mtmq_path_cnt.append(mtmq)
        mqmq_path_cnt.append(mqmq)
        mtq_path_cnt.append(mtq)
    
    np.save("features/" + train_or_test + "/mqtq_path_cnt.npy", np.array([mqtq_path_cnt]))
    np.save("features/" + train_or_test + "/mtmq_path_cnt.npy", np.array([mtmq_path_cnt]))
    np.save("features/" + train_or_test + "/mqmq_path_cnt.npy", np.array([mqmq_path_cnt]))
    np.save("features/" + train_or_test + "/mtq_path_cnt.npy", np.array([mtq_path_cnt]))
```


```python
# 建图
gb = GraphBuilder()

# 抽取问题类关联度特征
extractQ2QCorrelation("final")

# 抽取路径模式特征
extractPathPattern("final")
```


```python
"""
组合特征
"""

def combineFeatures(features_name, train_or_test):
    comb_fea = []
    file_path = "features/"+train_or_test+"/"
    features2columns = {
        "invite_week": ["invite_week"+str(i) for i in range(7)],
        "invite_hour": ["invite_hour"+str(i) for i in range(24)],
        "nn_inner": ["nn_inner"+str(i) for i in range(32)],
    }
    columns_name = []
    for fea in features_name:
        columns_name += features2columns[fea] if fea in features2columns else [fea]
    features = np.concatenate([np.load(file_path+fea+".npy").T for fea in features_name], axis=1)
    
    # fqy的特征
    print("fqy的特征...")
    fqy_data = pd.DataFrame(features, columns=columns_name)
    comb_fea.append(fqy_data)
    
    # dh的word_ctr特征
    print("dh的word_ctr特征...")
    word_ctr = pd.read_csv("../data/"+train_or_test+"/w_ctr.csv")
    comb_fea.append(word_ctr)
    
    # dh距离类的特征
    print("dh距离类的特征...")
    t2his_t_dis = pd.read_csv("../dh/feature/"+train_or_test+"/t2his_t_dis.csv",compression='gzip')
    t2his_a_dis = pd.read_csv("../dh/feature/"+train_or_test+"/t2his_a_dis.csv",compression='gzip')
    t2his_t_dis_weight = pd.read_csv("../dh/feature/"+train_or_test+"/t2his_t_dis_weight.csv",compression='gzip')
    dw_dis_v7 = pd.read_csv("../dh/feature/"+train_or_test+"/dw_dis_v7.csv")
    comb_fea.append(t2his_t_dis)
    comb_fea.append(t2his_a_dis)
    comb_fea.append(t2his_t_dis_weight)
    comb_fea.append(dw_dis_v7)
    
    # 合并数据及特征
    print("合并数据...")
    comb_data = pd.concat(comb_fea, axis=1)
    
    # 保存文件
    print("写入数据...")
    comb_data.to_csv("../data/"+train_or_test+"/fqy_dh_final_fea.txt", sep='\t', index=None, compression="gzip")
    
    pd.set_option('display.max_rows', None)
    return comb_data.head(100)


features_name = []
features_name += ["clf_a","clf_b","clf_c","clf_d","clf_e","gender_male","gender_female","gender_unknown",
                  "visit_freq_daily","visit_freq_weekly","visit_freq_monthly","visit_freq_new","visit_freq_unknown",
                  "salt_value","att_topic_num","int_topic_num"] # base √
features_name += ["qtitle_sw_cnt","qdesc_sw_cnt", "qtopic_cnt"] # √
features_name += ["date_invite2create", "cur_hour_answer_rate", "cur_week_answer_rate"] # √
features_name += ["q_inv_all", "q_inv_before", "q_inv_round", "q_inv_freq", "q_inv_round_freq", 
                  "q_first_inv", "q_last_inv", "q_inv_time_span", "q_inv_sametime_num"] # √
features_name += ["m_inv_before", "m_acpt_inv_before", "m_acpt_rate_before", "m_last_inv", "m_last_acpt_inv"] # √
features_name += ["m_acpt_inv_all", "m_acpt_rate_all"] # √
features_name += ["m_inv_sametime_num","m_inv_sameday_num","m_inv_samehour_num"] # √
features_name += ["m_ans_num","m_ans_like_sum","m_ans_cmt_sum","m_ans_pic_sum","m_ans_thx_sum","m_ans_clc_sum",
                  "m_ans_sw_sum","m_ans_sw_mean"] # √
features_name += ["m_last_ans", "m_first_ans", "m_ans_time_span", "m_ans_freq_mean", "m_ans_freq_std"] # √
features_name += ["q2q_corr_sum", "q2q_corr_len", "q2q_corr_mean"] # √
features_name += ["m_att_dist", "q_att_dist", "m_int_dist", "q_int_dist"] # √
features_name += ["mqtq_path_cnt","mtmq_path_cnt","mqmq_path_cnt","mtq_path_cnt"] # √
features_name += ["q_q_density_sum","q_q_density_mean","q_q_density_median","q_q_density_max","q_q_density_min"]
features_name += ["q_m_att_density_sum","q_m_att_density_mean","q_m_att_density_median","q_m_att_density_max","q_m_att_density_min"]
features_name += ["q_m_int_density_sum","q_m_int_density_mean","q_m_int_density_median","q_m_int_density_max","q_m_int_density_min"]
features_name += ["m_m_att_density_sum","m_m_att_density_mean","m_m_att_density_median","m_m_att_density_max","m_m_att_density_min"]
features_name += ["m_m_int_density_sum","m_m_int_density_mean","m_m_int_density_median","m_m_int_density_max","m_m_int_density_min"]
features_name += ["m_q_att_density_sum","m_q_att_density_mean","m_q_att_density_median","m_q_att_density_max","m_q_att_density_min"]
features_name += ["m_q_int_density_sum","m_q_int_density_mean","m_q_int_density_median","m_q_int_density_max","m_q_int_density_min"]

combineFeatures(features_name, "final")
```

### 问题与用户的距离类特征计算
&ensp;&ensp;&ensp;&ensp;sentence_vector 函数是计算句向量。calDis_t2t函数和calDis_t2a函数分别是根据用户历史回答问题的标题和回答内容来表征当前用户，然后与目标问题的标题计算距离


```python
#获取词向量
from gensim.models import KeyedVectors

wv_word_model=gensim.models.KeyedVectors.load_word2vec_format("./data/word_vectors_64d.txt",binary=False,
                                                              unicode_errors='ignore',
                                                               encoding="utf-8")
#计算句向量
def sentence_vector(words):
    v = np.zeros(64)
    i = 0
    for word in words:
        try:
            v += wv_word_model[word]
        except:
            i+=1
    if len(words)-i ==0:
        return v
    v /= (len(words)-i)
    return v
```


```python

# 获得问题id到标题的映射关系
answer_info =pd.read_csv('../data/origin/answer_info_0926.txt',header= None, sep = "\t", names = config_answer)
print(answer_info.shape)
answer_info_qid_list = answer_info["Q_id"].unique().tolist()

mid2answer = {}

for index, row in tqdm(answer_info.iterrows() ,total=answer_info.shape[0]):
    mid = row["M_id"]
    qid = row["Q_id"]
    aid = row["A_id"]
    date = int(row["A_create_time"].split("-")[0][1:])
    hour = int(row["A_create_time"].split("-")[1][1:])
    data=[(date, hour), [qid, aid]]
    if mid in mid2answer:
        mid2answer[mid].append(data)
    else:
        mid2answer[mid] = [data]
        
        
# 获得回答id到回答内容的映射关系
answer_info =pd.read_csv('../data/origin/answer_info_0926.txt',header= None, sep = "\t", names = config_answer)
print(answer_info.shape)
aid2TitleSenVec = {}
for index, row in tqdm(answer_info.iterrows() ,total=answer_info.shape[0]):
    answer = row['A_W_list'].strip().split(",")
    aid = row['A_id']
    if answer[0]!="-1":
        aid2TitleSenVec[aid] = sentence_vector(answer)        
print(len(aid2TitleSenVec))

del answer_info
gc.collect()
```


```python
# 相关度函数
import numpy as np
import math
from scipy.linalg import norm
import gensim
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

def Euclidean(vec1, vec2): ## 欧式
    npvec1, npvec2 = np.array(vec1), np.array(vec2)
    return math.sqrt(((npvec1-npvec2)**2).sum())

def Manhattan(vec1, vec2):
    npvec1, npvec2 = np.array(vec1), np.array(vec2)
    return np.abs(npvec1-npvec2).sum()

def wmdDistance(wv_model, s1, s2):
    wmd = wv_model.wv.wmdistance(s1, s2)
    if wmd == float("inf"):
        wmd = 700
    return wmd

def calSimilarity(wv_model,s1, s2):#余弦相似度【-1.1】 归一化【0，1】
    sim = wv_model.wv.similarity(s1, s2)
    return 0.5+ 0.5* sim

def cos_dist(vec1,vec2):
    vec1, vec2 = np.array(vec1), np.array(vec2)
    dist1=float(np.dot(vec1,vec2)/(np.linalg.norm(vec1)*np.linalg.norm(vec2)))
    return 0.5 * dist1 + 0.5

```


```python
# 计算t2t特征
def calDis_t2t(his_ans_qid_list, target_qid,qid2TitleSenVec):
    valid_qVec_list = []

    euc_list = []
    man_list = []
    cos_list = []

    flag, tar_vec = getSentenceVec(target_qid,qid2TitleSenVec)
    if not flag:
        res = [np.NAN] * 18
        return res

    for qid in his_ans_qid_list:
        flag, vec = getSentenceVec(qid,qid2TitleSenVec)
        if flag:
            valid_qVec_list.append(vec)

            euc_list.append(Euclidean(tar_vec, vec))
            man_list.append(Manhattan(tar_vec, vec))
            cos_list.append(cos_dist(tar_vec, vec))

    total_euc = np.NAN
    total_man = np.NAN
    total_cos = np.NAN

    res = []

    if len(valid_qVec_list) > 0:
        all_vec = np.mean(valid_qVec_list, axis=0)
        total_euc = Euclidean(tar_vec, all_vec)
        total_man = Manhattan(tar_vec, all_vec)
        total_cos = cos_dist(tar_vec, all_vec)

        res.append(total_euc)
        res.append(total_man)
        res.append(total_cos)
        for dis_list in [euc_list, man_list, cos_list]:
            res.append(np.mean(dis_list))
            res.append(np.max(dis_list))
            res.append(np.min(dis_list))
            res.append(np.median(dis_list))
            res.append(np.std(dis_list))
    else:
        res = [np.NAN] * 18

    return res
```


```python
# 计算t2a特征

def calDis_t2a(his_ans_aid_list, target_qid, qid2TitleSenVec, aid2TitleSenVec):
    valid_aVec_list = []

    euc_list = []
    man_list = []
    cos_list = []

    flag, tar_vec = getSentenceVec(target_qid, qid2TitleSenVec)
    if not flag:
        res = [np.NAN] * 18
        return res

    for aid in his_ans_aid_list:
        flag, vec = getSentenceVec(aid, aid2TitleSenVec)
        if flag:
            valid_aVec_list.append(vec)

            euc_list.append(Euclidean(tar_vec, vec))
            man_list.append(Manhattan(tar_vec, vec))
            cos_list.append(cos_dist(tar_vec, vec))

    total_euc = np.NAN
    total_man = np.NAN
    total_cos = np.NAN

    res = []

    if len(valid_aVec_list) > 0:
        all_vec = np.mean(valid_aVec_list, axis=0)
        total_euc = Euclidean(tar_vec, all_vec)
        total_man = Manhattan(tar_vec, all_vec)
        total_cos = cos_dist(tar_vec, all_vec)

        res.append(total_euc)
        res.append(total_man)
        res.append(total_cos)
        for dis_list in [euc_list, man_list, cos_list]:
            res.append(np.mean(dis_list))
            res.append(np.max(dis_list))
            res.append(np.min(dis_list))
            res.append(np.median(dis_list))
            res.append(np.std(dis_list))
    else:
        res = [np.NAN] * 18

    return res
```

### 词转化率类特征
&ensp;&ensp;&ensp;&ensp;get_WCTR_feature函数实现每个qid的词ctr的特征抽取


```python
# 计算词的转化率
w2ctr={}

def recordFun(qid, label, kFold):
    try:
        sentence = qid2Title[qid]
        
        for word in sentence:
            try:
                temp = w2ctr[word]
            except:
                w2ctr[word] = {}
            
            try:
                temp = w2ctr[word][kFold]
            except:
                w2ctr[word][kFold] = [0,0,0]
            
            w2ctr[word][kFold][0] += 1
            if label == 1:
                w2ctr[word][kFold][1] += 1
    except:
        pass

train_data =pd.read_csv('../data/train/train_data.txt')
print(train_data.shape)
print(train_data.columns)

for index, row in tqdm(train_data.iterrows() ,total=train_data.shape[0]):
    qid = row['Q_id']
    label = row['Label']
    kFold = row['5_fold']
    
    recordFun(qid, label, kFold)
    
def cal_log(click, app):
    return np.log(1 + (click / (1+app)))
```


```python
def get_WCTR_feature(qid, kFold):
    try:
        sentence = qid2Title[qid]
    except:
        return [np.NAN]*15
    
    app_list = []
    click_list = []
    ctr_list = []
        
    for word in sentence:
        if word not in w2ctr:
            continue
        
        temp = w2ctr[word][kFold]
        app_list.append(temp[0])
        click_list.append(temp[1])
        ctr_list.append(temp[2])
    
    if len(app_list) == 0:
        return [np.NAN]*15
    
    res = []
        
    for dis_list in [app_list, click_list,ctr_list]:
        res.append(np.mean(dis_list))
        res.append(np.max(dis_list))
        res.append(np.min(dis_list))
        res.append(np.median(dis_list))
        res.append(np.std(dis_list))
    return res
```

### Topic特征：
    一开始，针对问题和用户，将官方给的topic embeding,计算各种距离，来衡量相似度，但效果都很差。
    EDA后，发现topic有10W个取值，太过离散，于是将topic，重新聚类成百级别。 


```python
import pandas as pd
import numpy as np
config_topic =['T_id','T_embeding']
data =pd.read_csv('../data/origin/topic_vectors_64d.txt',sep='\t',header=None,names=config_topic)
topc_data = data['T_embeding']
topic_list = []
for t in topc_data:
    topic_list.append([float(x) for x in t.split(' ')])
from sklearn.cluster import KMeans
model = KMeans(n_clusters = 100, max_iter = 1000) 
model.fit(topic_list) 
r1 = pd.Series(model.labels_).value_counts() #统计各个类别的数目
r = pd.DataFrame(model.cluster_centers_) #找出聚类中心
count = [i for i in range(50)]
types = ['Embeding'+str(i) for i in range(64)]
r.columns =  list(types) 
r['T_type']=count
cluster_topic = pd.concat([data['T_id'], pd.Series(model.labels_, index = data.index)], axis = 1)
cluster_topic.columns = ['T_id','T_type']
cluster_topic = pd.merge(test,r,how='left',on=['T_type'])
cluster_topic.to_csv('cluster_topic.csv',index=False)
```

利用K-means将topic聚类后，发现每个问题和用户的多个topic标签基本被聚到一个簇，根据question的topic和用户的interest topic重新生成新的topic


```python
topic_dict = {}
topic_cluster = pd.read_csv('cluster_topic.csv')
for i,r in topic_cluster.iterrows():
    topic_dict[r['T_id']]=r['T_type']
    
config_question = ['Q_id','create_time','QTitle_SW_list','QTitle_W_list','QDescrible_SW_list','QDescrible_W_list','QT_list']
config_member = ['M_id','Gender','Create_KeyW','Create_Count','Create_hot','Reg_Type','Reg_Platform','Frequency','Feature_A','Feature_B','Feature_C','Feature_D','Feature_E','Feature_a','Feature_b','Feature_c','Feature_d','Feature_e','Score','T_attention','T_interest']
q_data = pd.read_csv('../data/origin/question_info_0926.txt',sep='\t',header=None,names=config_question)
m_data = pd.read_csv('../data/origin/member_info_0926.txt',sep='\t',header=None,names=config_member)
q_data = pd.DataFrame(q_data,columns=['Q_id','QT_list'])
m_data = pd.DataFrame(m_data,columns=['M_id','T_interest'])

q_cluster = pd.DataFrame(q_data,columns=['Q_id'])
qt_cluster = []
for i,r in q_data.iterrows():
    res = r['QT_list'].split(',')[0]
    if res=='-1':
        qt_cluster.append(-1)
    else:
        qt_cluster.append(topic_dict[r['QT_list'].split(',')[0]])
q_cluster['qt_cluster']= qt_cluster

m_cluster = pd.DataFrame(m_data,columns=['M_id'])
qm_cluster = []
for i,r in m_data.iterrows():
    res = r['T_interest'].split(',')[0].split(':')[0]
    if res=='-1':
        qm_cluster.append(-1)
    else:
        qm_cluster.append(topic_dict[res])
m_cluster['mt_cluster']= qm_cluster

q_cluster.to_csv('cluster_qtopic.csv',index=False)
m_cluster.to_csv('cluster_mtopic.csv',index=False)
```

根据邀请表，为每一个邀请的问题和用户赋予新的topic，并计算topic级别的相关距离特征


```python
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn import metrics
import operator


data1 =pd.read_csv('../data/final/test.txt')
data1 = pd.DataFrame(data1,columns=['Q_id','M_id','Invite_Time'])
data_m = pd.read_csv('cluster_mtopic.csv')
data_m = data_m.drop_duplicates()
data_q = pd.read_csv('cluster_qtopic.csv')
data_q = data_q.drop_duplicates()
data1 = pd.merge(data1,data_m,how='left',on=['M_id'])
data1 = pd.merge(data1,data_q,how='left',on=['Q_id'])
data2 =pd.read_csv('cluster_topic.csv')
topic_dict = {}
for i in range(64):
    data2['Embeding'+str(i)] = round(data2['Embeding'+str(i)],4)
for i,r in data2.iterrows():
    if r['T_type'] not in topic_dict.keys():
        topic_dict[r['T_type']] = [r['Embeding'+str(i)] for i in range(64)]
        
def Euclidean(vec1, vec2):
    npvec1, npvec2 = np.array(vec1), np.array(vec2)
    return math.sqrt(((npvec1-npvec2)**2).sum())

def Manhattan(vec1, vec2):
    npvec1, npvec2 = np.array(vec1), np.array(vec2)
    return np.abs(npvec1-npvec2).sum()
# Manhattan_Distance,曼哈顿距离


def Chebyshev(vec1, vec2):
    npvec1, npvec2 = np.array(vec1), np.array(vec2)
    return max(np.abs(npvec1-npvec2))

def cosine_similarity(x, y, norm=False):
    """ 计算两个向量x和y的余弦相似度 """
    assert len(x) == len(y), "len(x) != len(y)"
    zero_list = [0] * len(x)
    if x == zero_list or y == zero_list:
        return float(1) if x == y else float(0)

    res = np.array([[x[i] * y[i], x[i] * x[i], y[i] * y[i]] for i in range(len(x))])
    cos = sum(res[:, 0]) / (np.sqrt(sum(res[:, 1])) * np.sqrt(sum(res[:, 2])))

    return 0.5 * cos + 0.5 if norm else cos  # 归一化到[0, 1]区间内
mt_topic_dif = []
mt_topic_Euclidean = []
mt_topic_Manhattan = []
mt_topic_Chebyshev = []
mt_topic_cosine_similarity = []
for i,r in data1.iterrows():
    if i%100000 ==0:
        print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
        print(i)
    if r['mt_cluster'] ==r['qt_cluster']:
        mt_topic_dif.append(0)
    else:
        mt_topic_dif.append(1)
    if r['mt_cluster'] ==-1 or r['qt_cluster']==-1:
        mt_topic_Euclidean.append(-1)
        mt_topic_Manhattan.append(-1)
        mt_topic_Chebyshev.append(-1)
        mt_topic_cosine_similarity.append(-1)
    else:
        mt_topic_Euclidean.append(Euclidean(topic_dict[r['mt_cluster']],topic_dict[r['qt_cluster']]))
        mt_topic_Manhattan.append(Manhattan(topic_dict[r['mt_cluster']],topic_dict[r['qt_cluster']]))
        mt_topic_Chebyshev.append(Chebyshev(topic_dict[r['mt_cluster']],topic_dict[r['qt_cluster']]))
        mt_topic_cosine_similarity.append(cosine_similarity(topic_dict[r['mt_cluster']],topic_dict[r['qt_cluster']]))
data1['mt_topic_dif'] = mt_topic_dif
data1['mt_topic_Euclidean'] = mt_topic_Euclidean
data1['mt_topic_Manhattan'] = mt_topic_Manhattan
data1['mt_topic_Chebyshev'] = mt_topic_Chebyshev
data1['mt_topic_cosine_similarity'] = mt_topic_cosine_similarity
```

根据新的topic，计算topic的相关统计特征，如邀请数、回答率、回答数等

## LGB部分
将所提取的231维特征灌入LGB训练，A榜单模0.8632


```python
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn import metrics
import operator
import gc
print ('start')
data1 =pd.read_csv('data/final/fqy_dh_final_fea.txt')
data1 = pd.DataFrame(data1,columns=['Q_id','M_id','Invite_Time','Label'])
data_3 = pd.read_csv('zhongr/topic_distance.csv')
data_3 = data_3.drop_duplicates()
data_4 = pd.read_csv('zhongr/topic_count_train.csv')
data_4 = data_4.drop_duplicates()
data_5 = pd.read_csv('Q_embding.txt')

print(data1.shape)
data1 = pd.merge(data1,data_3,how='left',on=['Q_id','M_id','Invite_Time'])
data1 = pd.merge(data1,data_4,how='left',on=['Q_id','M_id','Invite_Time'])
data1 = pd.merge(data1,data_5,how='left',on=['Q_id'])
print(data1.shape)
data2 = pd.read_csv("data/train/fqy_11_22.txt", compression='gzip',sep='\t')
print (data1.shape,data2.shape)
train_data = pd.concat([data1,data2],axis=1)
train_label = train_data['Label']

from sklearn.model_selection import train_test_split
train_data,val_data,train_label,val_label = train_test_split(train_data,train_label,test_size=0.2,random_state=0)
train_data.drop(columns=['Q_id', 'M_id', 'Invite_Time', 'Label'],axis=1,inplace=True)
val_data.drop(columns=['Q_id', 'M_id', 'Invite_Time', 'Label'],axis=1,inplace=True)
print (train_data.shape,val_data.shape)
print (train_data.columns.values)
print("start")

trn_data = lgb.Dataset(train_data,
                       label=train_label
                       )

vals_data = lgb.Dataset(val_data,
                        label=val_label
                        )

param = {
    'max_depth': 7,
    'num_leaves':40,
    'objective': 'binary',
    #'num_leaves': 32,
    #'num_trees': 1000,
    'bagging_fraction': 0.7,
    'feature_fraction': 0.7,
    'learning_rate': 0.02,
    "boosting": "gbdt",
    "bagging_freq": 5,,
    "metric": 'auc',#auc binary_logloss
    "nthread": 16,
    "verbosity": -1
}
clf = lgb.train(param,
                train_set=trn_data,
                valid_sets=vals_data,
                num_boost_round=100000,
                early_stopping_rounds=700,
                verbose_eval=100)
clf.save_model('lgb_1215.model')
val_pre = clf.predict(val_data, num_iteration=clf.best_iteration)

print("auc score: {:<8.5f}".format(metrics.roc_auc_score(val_label, val_pre)))

print(pd.DataFrame({
        'column': val_data.columns.values,
        'importance': clf.feature_importance()
    }).sort_values(by='importance'))
```

## nn部分
&ensp;&ensp;&ensp;&ensp; 主要使用了xdeepfm网络结构对特征做交叉计算，此块代码主要基于DeepCTR的实现做了修改，git地址为https://github.com/shenweichen/DeepCTR
&ensp;&ensp;&ensp;&ensp;因此部分代码可在deepctr的git中找到，故不在此贴出。A榜单模0.8623


```python
class DotProductAttention(Layer):
    def __init__(self, return_attend_weight=False, keep_mask=True, **kwargs):
        self.return_attend_weight = return_attend_weight
        self.keep_mask = keep_mask
        self.supports_masking = True
        super(DotProductAttention, self).__init__(**kwargs)

    def build(self, input_shape):
        assert isinstance(input_shape, list)
        input_shape_a, input_shape_b = input_shape

        if len(input_shape_a) != 3 or len(input_shape_b) != 3:
            raise ValueError('Inputs into DotProductAttention should be 3D tensors')

        if input_shape_a[-1] != input_shape_b[-1]:
            raise ValueError('Inputs into DotProductAttention should have the same dimensionality at the last axis')

    def call(self, inputs, mask=None):
        assert isinstance(inputs, list)
        inputs_a, inputs_b = inputs

        if mask is not None:
            mask_a, mask_b = mask
        else:
            mask_a, mask_b = None, None

        e = K.exp(K.batch_dot(inputs_a, inputs_b, axes=2))  # similarity between a & b

        if mask_a is not None:
            e *= K.expand_dims(K.cast(mask_a, K.floatx()), 2)
        if mask_b is not None:
            e *= K.expand_dims(K.cast(mask_b, K.floatx()), 1)

        e_b = e / K.cast(K.sum(e, axis=2, keepdims=True) + K.epsilon(), K.floatx())  # attention weight over b
        e_a = e / K.cast(K.sum(e, axis=1, keepdims=True) + K.epsilon(), K.floatx())  # attention weight over a

        if self.return_attend_weight:
            return [e_b, e_a]

        a_attend = K.batch_dot(e_b, inputs_b, axes=(2, 1))  # a attend to b
        b_attend = K.batch_dot(e_a, inputs_a, axes=(1, 1))  # b attend to a
        return [a_attend, b_attend]

    def compute_mask(self, inputs, mask=None):
        if self.keep_mask:
            return mask
        else:
            return [None, None]

    def compute_output_shape(self, input_shape):
        if self.return_attend_weight:
            input_shape_a, input_shape_b = input_shape
            return [(input_shape_a[0], input_shape_a[1], input_shape_b[1]),
                    (input_shape_a[0], input_shape_a[1], input_shape_b[1])]
        return input_shape


```


```python
def buildModel(SparseFeat_valCount,sparse_features, dense_features,wv_embed_matrix):

    # Define the model
    print("Define the model")
    print("***********************")
    fixlen_feature_columns = [SparseFeat(feat, SparseFeat_valCount[feat], use_hash=False,dtype="int32") 
                              for feat in sparse_features] +  [DenseFeat(feat, 1, ) for feat in dense_features]
    
    
    linear_feature_columns = fixlen_feature_columns 
    dnn_feature_columns = fixlen_feature_columns
    
    # input
    featuresInputLayer_dict = build_input_features(fixlen_feature_columns)
    print("***********************")
    inputLayers_list = list(featuresInputLayer_dict.values())
    inputLayerNames_list = list(featuresInputLayer_dict.keys())
    
    sparse_embedding_list, dense_value_list = input_from_feature_columns(featuresInputLayer_dict,
                                                                         dnn_feature_columns,
                                                                         embedding_size,0.00001, 0.0001, 1)
    print("***********************")
    #lr 
    linear_logit = get_linear_logit(sparse_embedding_list, dense_value_list , 
                                    l2_reg=0.00001, init_std=0.0001,
                                    seed=1, prefix='linear')
    print("***********************")
    # 二次项
    fm_input = concat_fun(sparse_embedding_list, axis=1)
    fm_logit = FM()(fm_input)
    print("***********************")
    
    #dnn
    dnn_input = combined_dnn_input(sparse_embedding_list,dense_value_list)
    dnn_out = DNN((256, 128, 64), 'relu', 0.00001, 0.2,
                   False, 1)(dnn_input)
    print("***********************")
    dnn_logit = tf.keras.layers.Dense(
        1, use_bias=False, activation=None)(dnn_out)
    
    dense_fm_out = FMLayer(256,50,activation='relu')(Flatten()(concat_fun(dense_value_list)))
    dense_fm_out = Dropout(0.3)(dense_fm_out)
    dense_fm_out = BatchNormalization()(dense_fm_out)
    
    dense_fm_logit = tf.keras.layers.Dense(
        1, use_bias=False, activation=None)(dense_fm_out)
    
    
    dense_emb = []
    for dense_val in dense_value_list:
        dense_emb.append(RepeatVector(1)(tf.keras.layers.Dense(embedding_size, use_bias=False)(dense_val)))
    
    all_emb = dense_emb
    all_emb.extend(sparse_embedding_list)
    all_emb =  Concatenate(axis=1)(all_emb)
    
    #CIN
    exFM_out = CIN((128, 128,), 'relu', True, 0, 1)(all_emb)
    exFM_logit = tf.keras.layers.Dense(1, activation=None, )(exFM_out)
    
    target_title_input = Input(shape=(max_seq_len,), dtype='int32',name="target_title_input")
    his1_title_input = Input(shape=(max_seq_len,), dtype='int32',name="his1_title_input")
    his2_title_input = Input(shape=(max_seq_len,), dtype='int32',name="his2_title_input")
    his3_title_input = Input(shape=(max_seq_len,), dtype='int32',name="his3_title_input")
    
    wv_embed_layer = Embedding(wv_embed_matrix.shape[0], embed_dim, weights=[wv_embed_matrix],
                                input_length=max_seq_len, trainable=False)
    
    target_title = wv_embed_layer(target_title_input)
        
    his_title = []
    his_title.append(wv_embed_layer(his1_title_input))
    his_title.append(wv_embed_layer(his2_title_input))
    his_title.append(wv_embed_layer(his3_title_input))
    

    shared_lstm_1 =Bidirectional(CuDNNLSTM(64,return_sequences=True))
    feed_forward = TimeDistributed(Dense(units=64, activation='relu'))
    shared_bilstm_2 = CuDNNLSTM(units=64, return_sequences=True)
    global_max_pooling = Lambda(lambda x: K.max(x, axis=1))

    #lstm
    lstm_target_hidden = shared_lstm_1(target_title)
    lstm_res = []
    for ht in his_title:
        lstm_his_hidden = shared_lstm_1(ht)
        tar_attend, his_attend = DotProductAttention()([lstm_target_hidden, lstm_his_hidden])
        tar_enhance = concatenate([lstm_target_hidden, tar_attend, subtract([lstm_target_hidden, tar_attend]),
                                     multiply([lstm_target_hidden, tar_attend])])  
        his_enhance = concatenate([lstm_his_hidden, his_attend,
                                     subtract([lstm_his_hidden, his_attend]),
                                     multiply([lstm_his_hidden, his_attend])]) 

        tar_compose = shared_bilstm_2(feed_forward(tar_enhance))
        his_compose = shared_bilstm_2(feed_forward(his_enhance))
        tar_avg = GlobalAveragePooling1D()(tar_compose)
        tar_max = global_max_pooling(tar_compose)
        his_avg = GlobalAveragePooling1D()(his_compose)
        his_max = global_max_pooling(his_compose)

        inference_compose = concatenate([tar_avg, tar_max,his_avg,his_max])
        inference_compose = BatchNormalization()(inference_compose) 
        dense_esim = Dense(units=64)(inference_compose)
        
        lstm_res.append(dense_esim)
    
    
    lstm_merge = concatenate(lstm_res)
    lstm_merge = Flatten()(lstm_merge)
    lstm_merge = BatchNormalization()(lstm_merge)
    lstm_merge = Dense(128,activation='relu',kernel_initializer="lecun_uniform")(lstm_merge) 
    lstm_merge = Dropout(0.3)(lstm_merge)
    lstm_merge = BatchNormalization()(lstm_merge)
    lstm_merge = Dense(64,activation='relu',kernel_initializer="lecun_uniform")(lstm_merge) 
    
    lstm_logit =  tf.keras.layers.Dense(1, use_bias=False, activation=None)(lstm_merge) 
    
    #all_merge
    all_merge = concatenate([lstm_merge,dense_fm_out, dense_fm_out, linear_logit,fm_logit,exFM_out])
    all_merge = Flatten()(all_merge)
    all_merge = concatenate([Flatten()(concat_fun(dense_value_list)),all_merge])
    all_merge = BatchNormalization()(all_merge)
    
    all_merge = Dense(256,activation='relu',kernel_initializer="lecun_uniform")(all_merge) 
    all_merge = Dropout(0.3)(all_merge)
    all_merge = BatchNormalization()(all_merge)
    
    all_merge = Dense(128,activation='relu',kernel_initializer="lecun_uniform")(all_merge) 
    all_merge = Dropout(0.3)(all_merge)
    all_merge = BatchNormalization()(all_merge)
    
    all_merge = Dense(64,activation='relu',kernel_initializer="lecun_uniform")(all_merge) 

    all_merge_logit =  tf.keras.layers.Dense(1, use_bias=False, activation=None)(all_merge) 

    #logit
    final_logit = tf.keras.layers.add([linear_logit, fm_logit, dnn_logit,dense_fm_logit,
                                       exFM_logit,lstm_logit,all_merge_logit])
    output = PredictionLayer("binary")(final_logit)
    
    inputLayers_list.append(target_title_input)
    inputLayers_list.append(his1_title_input)
    inputLayers_list.append(his2_title_input)
    inputLayers_list.append(his3_title_input)
    
    model = tf.keras.models.Model(inputs=inputLayers_list, outputs=output)
    model.compile(optimizer=Adam(lr=0.0002), 
                  loss="binary_crossentropy",
                  metrics=['binary_crossentropy','accuracy',auc])
    
    print(model.summary())
  
    return model
```

## 模型融合
该部分将LGB与nn模型进行加权融合，由上面两部分预测结果构成，代码不再贴出。融合结果：A榜0.868+，B榜：0.8872
