# Machine Learning Project (MNIST)
Machine Learning group project.

## Proposal
Proposal需要包含以下部分：
> The project proposal should be at most one page with the following contents: 
> 1) an introduction that briefly states the problem; 
> 2) a precise description of what you plan to do – e.g., What types of features do you plan to use? What algorithms do you plan to use? What dataset will you use? How will you evaluate your results? How do you define a good outcome for the project? 
> The goal of the proposal is to work out, in your head, what your project will be. Once the proposal is done, it is just a matter of implementation!

### Introduction
研究不同machine learning方法在mnist数据集上的准确率。（待想）

### Detailed Plan
参考Methodology部分。



## Methodology
General idea: 使用不同模型对数据进行训练，比较准确率。
* 需要哪些模型？二分类或十分类？
* 数据的预处理部分（normalization, dimension reduction, transformation, etc.）
* ...

## Dataset
MNIST数据集（不完整）
* Training data: (2000,786) digits, (2000,) labels.
* Test data: (2000,786) digits, (2000,) labels.
* 数据是都是按照0-9的顺序排列的，每个数字200个。

## Models
在model目录中有一些现有的模型，包括SVM，CNN，DNN等，目前准确率不高，方法待改进。




