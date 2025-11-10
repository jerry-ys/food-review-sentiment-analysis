#数据读取
import pandas as pd
reviews = pd.read_csv('Reviews.csv')
print(reviews.shape)
reviews.head()
reviews.drop('ProfileName',axis=1,inplace=True)
#缺失值处理
reviews.isnull().sum()
reviews.dropna(inplace=True)
print(reviews.shape)
#重复值处理,删去重复的文本，只保留第一个
print(reviews.duplicated(subset=['Text']).sum())
reviews.drop_duplicates(subset=['Text'],keep='first',inplace=True)
print(reviews.shape)
# 异常值处理
reviews[reviews["HelpfulnessNumerator"]>reviews["HelpfulnessDenominator"]]
reviews.drop([44736,64421],inplace=True)
# 查看剩余的异常值
reviews[reviews["HelpfulnessNumerator"]>reviews["HelpfulnessDenominator"]]
reviews.reset_index(drop=True,inplace=True)
 #时间类型数据进行处理
reviews['Time'] = pd.to_datetime(reviews['Time'],unit='s')
reviews['Time']
reviews['year'] = reviews['Time'].dt.year
reviews.head()
#文本数据处理
reviews['Text']
#将所有字母转换为小写字母
reviews["Text"]=reviews["Text"].str.lower()
reviews["Summary"] = reviews["Summary"].str.lower()

#删除非英文字符
import re
reviews['Text_cl'] = reviews['Text'].apply(lambda x: re.sub(r'[^a-z]+', ' ', x))
reviews['Summary'] = reviews['Summary'].apply(lambda x: re.sub(r'[^a-z]+', ' ', x))
#去停用词
import nltk
# nltk.download('stopwords')
from nltk.corpus import stopwords
sw = stopwords.words('english')
sw = sw+['br']
sw

reviews["Text_cl"] =reviews["Text_cl"].apply(lambda x:" ".join(x for x in str(x).split() if x not in sw or x == "not"))
reviews["Summary"] =reviews["Summary"].apply(lambda x:" ".join(x for x in str(x).split() if x not in sw or x == "not"))
reviews['Text_cl']
#构造评论有用性特征
import numpy as np
result = np.where(reviews['HelpfulnessDenominator']==0,np.nan,reviews['HelpfulnessNumerator']/reviews['HelpfulnessDenominator'])
reviews['Usefulness'] = np.where(result>0.5,'useful',np.where(np.isnan(result),'unknown','useless'))
reviews
