### 2018 CCF大数据竞赛

`网址`：https://www.datafountain.cn/competitions/311/details/rule

#### 赛题名称：


`面向电信行业存量用户的智能套餐个性化匹配模型`


#### 赛题背景：
`电信产业作为国家基础产业之一，覆盖广、用户多，在支撑国家建设和发展方面尤为重要。随着互联网技术的快速发展和普及，用户消耗的流量也成井喷态势，近年来，电信运营商推出大量的电信套餐用以满足用户的差异化需求，面对种类繁多的套餐，如何选择最合适的一款对于运营商和用户来说都至关重要，尤其是在电信市场增速放缓，存量用户争夺愈发激烈的大背景下。针对电信套餐的个性化推荐问题，通过数据挖掘技术构建了基于用户消费行为的电信套餐个性化推荐模型，根据用户业务行为画像结果，分析出用户消费习惯及偏好，匹配用户最合适的套餐，提升用户感知，带动用户需求，从而达到用户价值提升的目标。
套餐的个性化推荐，能够在信息过载的环境中帮助用户发现合适套餐，也能将合适套餐信息推送给用户。解决的问题有两个：信息过载问题和用户无目的搜索问题。各种套餐满足了用户有明确目的时的主动查找需求，而个性化推荐能够在用户没有明确目的的时候帮助他们发现感兴趣的新内容。`

#### 赛题任务：


此题利用已有的用户属性(如个人基本信息、用户画像信息等)、终端属性(如终端品牌等)、业务属性、消费习惯及偏好匹配用户最合适的套餐，对用户进行推送，完成后续个性化服务。

#由于是新手，基本上就按照以下步骤进行的：

   - [x] 数据分析，主要是利用第三方matplotlib库，利用图示的方式来对数据进行展示
   
   - [x] 数据清洗
   
   - [x] 数据特征提取
   
   - [x] 对于一些缺失值进行填充
   
   - [x] 由于数据集中既包含类别特征，也包含连续特征，所以对类别特征进行one-hot编码，对连续特征进行归一化操作。
   
   - [x] 利用随机森林和GBDT结合GridSearchCV()函数来对特征重要性进行排序，然后选择特征重要性比较高的特征,这样能去掉一些重要性较低的特征。
   
   - [x] 在数据特征工程做的差不多的时候，接下来进行模型的选择：
   
     (1)lgb模型  f1_score:0.7441
     
     (2)xgb模型  f1_score:0.7521
     
   - [x] 这里没有使用stacking方式，而是简单的将上述的两个模型通过调整不同的参数来生成多个结果，然后再将这些结果通过投票的原则生成一个最终的结果，最终的f1_score:0.7693
   
   
 #### 参考资料：
  - https://blog.csdn.net/github_38414650/article/details/76061893
  
  - https://blog.csdn.net/hqr20627/article/details/79426031
  
  - https://www.jianshu.com/p/48e82dbb142b
  
  - https://www.jianshu.com/p/5378ef009cae
  
  - https://blog.csdn.net/sinat_26917383/article/details/54667077

#### 初赛成绩：
|A榜|B榜|
|:----|:----|
|38/2546|49/2546|

