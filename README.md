这是一个学习项目，结合D:\desktop\quackDocs\my_notes\知识库\raw\personal-notes\编程\RAG一遍学习一遍进行

# 技术选型

1. 向量数据库选择：sqlite-vec 

   为什么：ifeprism使用的是sqlite，而且数据量并不算大，不需要ANN ，而且RAG并不是当前项目的主要功能，作为桌面应用扩展，相对于其他数据库需要开销小

2. 