#!/usr/bin/env python
# coding: utf-8

# In[43]:


get_ipython().run_line_magic('pip', 'install networkx')


# # Importing Libraries

# In[101]:


import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
import networkx as nx
import random


# # Making Dictionary

# In[45]:


words = {}
with open('text-2.txt') as f:
    for line in f:
        # removing All nan alphabetic and numeric characters
        edited_line =  re.sub(r'[^\w\s]', ' ', line)
        for word in edited_line.split():
            if word in words.keys():
                words[word] = words[word] + 1
            else:
                words[word] = 0 + 1
words


# # Making Dataframe

# In[46]:


df = pd.DataFrame(list(words.items()), columns=['word', 'frequency'])
df


# # Sorting Dataframe (Part1)

# In[47]:


df = df.sort_values("frequency", ascending=False)
df['rank'] = df['frequency'].rank(ascending=False)
df


# # Drawing Charts

# ### Simple Plots

# In[56]:


df.plot(x ='rank', y='frequency', kind='line')


# In[57]:


df.plot(x ='rank', y='frequency', kind='bar')


# In[64]:


df.plot(x ='frequency', y='rank', kind='bar')


# In[58]:


df.plot(x ='rank', y='frequency', kind='scatter')


# ### Double Log Plots

# In[62]:


#perform log transformation on both x and y
xlog = np.log(df['rank'].to_numpy())
print('xlog finished')
ylog = np.log(df['frequency'].to_numpy())
print('ylog finished')

#create log-log plot
plt.scatter(xlog, ylog)

# adding lables for plot
plt.xlabel('Log(x)')
plt.ylabel('Log(y)')
plt.title('Log-Log Plot')


# In[63]:


#create log-log plot
plt.plot(xlog, ylog)

# adding lables for plot
plt.xlabel('Log(x)')
plt.ylabel('Log(y)')
plt.title('Log-Log Plot')


# In[66]:


#create log-log plot
plt.bar(xlog, ylog)

# adding lables for plot
plt.xlabel('Log(x)')
plt.ylabel('Log(y)')
plt.title('Log-Log Plot')


# # Making Graphs

# ### Small World Graph

# In[75]:


words_number = len(df)
# if we want to have equal nodes and edges number,
# each node, must be connected to its 2 nearest neighbor

G = nx.watts_strogatz_graph(n=words_number, k=2, p=0.5)
# if we want a connected graph we should use this:
# G = nx.connected_watts_strogatz_graph(n=10, m=4, p=0.5, t=20)
pos = nx.circular_layout(G)
 
plt.figure(figsize = (12, 12))
nx.draw_networkx(G, pos)


# In[76]:


# Simple Example oF Correct Small World Graph
G = nx.watts_strogatz_graph(n=20, k=4, p=0.5)
# if we want a connected graph we should use this:
# G = nx.connected_watts_strogatz_graph(n=10, m=4, p=0.5, t=20)
pos = nx.circular_layout(G)
 
plt.figure(figsize = (12, 12))
nx.draw_networkx(G, pos)


# ### Random Graph

# ###### Random Graph With N Nodes & M Edges

# In[77]:


words_number = len(df)
G = nx.gnm_random_graph(words_number, words_number)

plt.figure(figsize = (12, 12))
nx.draw(G)


# In[86]:


# Simple Example oF Correct Random Graph
G = nx.gnm_random_graph(20, 40)

plt.figure(figsize = (7, 7))
nx.draw(G)


# ###### Random Graph With N Nodes & M% Edge Probability

# In[87]:


words_number = len(df)
G = nx.gnp_random_graph(words_number, words_number)

plt.figure(figsize = (12, 12))
nx.draw(G)


# In[97]:


# Simple Example oF Correct Random Graph
G = nx.gnp_random_graph(20, 0.35)

plt.figure(figsize = (7, 7))
nx.draw(G)


# ###### Random Graph With N Nodes With Degree M

# In[109]:


words_number = len(df)
m = random.randint(1, int(words_number/1000))
print(m)
G = nx.random_regular_graph(m, words_number)

plt.figure(figsize = (12, 12))
nx.draw(G)


# In[113]:


G = nx.random_regular_graph(8, 20)

plt.figure(figsize = (12, 12))
nx.draw(G)


# ### Scale Free Graph with N Nodes

# In[115]:


words_number = len(df)
G = nx.scale_free_graph(words_number)

plt.figure(figsize = (12, 12))
nx.draw(G)


# In[123]:


G = nx.scale_free_graph(10)

plt.figure(figsize = (7, 7))
nx.draw(G)

