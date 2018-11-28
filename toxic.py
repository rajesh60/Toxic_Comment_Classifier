#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# # This Python 3 environment comes with many helpful analytics libraries installed
# # It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# # For example, here's several helpful packages to load in 

# import numpy as np # linear algebra
# import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# # Input data files are available in the "../input/" directory.
# # For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

# import os
# print(os.listdir("../input"))

# # Any results you write to the current directory are saved as output.


# In[ ]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


data=pd.read_csv('../input/train.csv')
X=data['comment_text']
Y=data.iloc[:,2:9]




from sklearn.feature_extraction.text import TfidfVectorizer
tfVector = TfidfVectorizer(stop_words='english')


# In[ ]:


X_train=list(X)

tfVector.fit(X_train)

Xlist_train=tfVector.transform(X_train)


# In[ ]:


from sklearn.ensemble import RandomForestClassifier
reg = RandomForestClassifier(n_estimators=20, criterion='entropy', random_state=10)
reg.fit(Xlist_train,Y)


# In[ ]:


test=pd.read_csv('../input/test.csv')


# In[ ]:


test.head()


# In[ ]:


x_test=test['comment_text']


# In[ ]:


x_test.head()


# In[ ]:


xlist_test=tfVector.transform(x_test)


# In[ ]:


xlist_test[0].toarray()


# In[ ]:


pred=reg.predict(xlist_test)


# In[ ]:


pred[0]


# In[ ]:


pred[:,0]


# In[ ]:


submission=pd.DataFrame({
    "id":test['id'],"toxic":pred[:,0],"severe_toxic":pred[:,1],"obscene":pred[:,2],"threat":pred[:,3],"insult":pred[:,4],"identity_hate":pred[:,5]
})
submission.to_csv('submission.csv',index=False)


# In[ ]:




