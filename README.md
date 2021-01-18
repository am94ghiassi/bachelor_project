# Multi-Label Learning with Incomplete Information

## Supervision Team

Responsible Professor: Dr. Lydia Y. Chen

Supervisors: Amirmasoud Ghiassi and Taraneh Younesian

## Background and motivation

multi-label Learning is a extension of multi-class classification, which multiple labels are assigned to each sample. As shown in Figure~\ref{subfig:mml}, A set of labels is associated to each image. In Multi-Label Learning, the goal is to train Deep Neural Networks with multi-label examples[^1]. Acquiring a fully labeled dataset in multi-labeled scenario is a time-consuming and expensive task, therefore a big challenge in this area is to provide the clean label set.  
In many cases usually the provided label set is partially included with true labels named Partial Multi-Label Learning~\cite{xie2018partial}. An example of PML is shown in Figure~\ref{subfig:pml}. As seen in the figure with PML, not all the provided labels are true and only a subset of the given labels is true. Then the challenge is to identify the correct labels among the provided label set. A solution for this issue is to construct an example-label relevance confidence matrix and estimate it based on the available label information. The prior art~\cite{lyu2020partial, xie2018partial} selects high-confidence labels, then using the off-the-shelf multi-label learning method to train models in Partial Multi-Label Learning scenarios.      

Another challenge arising in multi-label learning, similar to single label learning, is incomplete label/data information. However, since there are more than one label solution per example in multi-labeled datasets, this tasks becomes more difficult to tackle. One example of the incompleteness is online learning, where all the data is not available ahead of time. In this case, data arrival is batch by batch and online and thus the algorithm needs to adapt to the dynamics of data arrival and train on every small batch of data.

Moreover, missing labels is another emerging topic in multi-label learning. This issue frequently happen in domains where the labels are collected via crowd-sourcing to reduce the labeling cost and effort. Furthermore, independent from crowd-sourcing, in some applications training based on incomplete label set and predicting the unavailable labels is preferred to reduce the annotation cost. This issue is illustrated in Figure \ref{subfig:mlmissing}, where only some of the true labels are available for the image. In these cases, the question that needs to be answered is "\textit{how to train an accurate multi-labeled classifier with the missing or incomplete information?}".

![Convolution](/home/bacox/Pictures/Convolution.png)
Figure 1: An illustration of a convolutional neural network, consisting of convolution, pooling andfully-connected layers which are known to have different resource characteristics.

**Specific problem scenario**: Images are constantly taken from camera and fed into the inference engine that needs to answer the questions about peoples' race, age, and gender. The inference engine can load pre-trained deep neural networks to answer those questions for each of the pictures. In other words, each picture represents a job and consists of three subtasks of identifying race, age, and gender. The inference engine is hosted on edge devices that have limited CPU power and memory space. The research questions here are how to best execute those jobs such that their response times are minimized at a minimal energy consumption. The edge devices considered here can be laptops or mobile phones. 


### Research Questions for the Sub-Projects

In this project, we aim to develop novel scheduling policies that can minimize the response time for executing multi-inference jobs, each of which is composed of multiple inference tasks using different trained networks. To such an end, the first step is to develop a good understanding of resource demands, e.g., CPU, memory, or IO, of different types of deep inference jobs and their respective layers. Take convolutional neural network as an example shown in Fig 1., a fully connected layer of the convolution network is memory intensive, whereas the convolutional layer is CPU intensive. The second step is to find the optimal way to partition those inference jobs according to their network characteristics. For instance, each inference task can be decomposed into multiple sub-tasks that correspond to each layer of neural networks. The third step is to combine the understanding of resource demands per layer and explore different scheduling policies.

![dnn_pool](/home/bacox/Pictures/dnn_pool_execution.png)
Figure 2: An illustration on how a scheduler can select layers from a pool of deep neural networks ona two-core system.

**Expected Novelty** The prior work mainly aims to optimize the execution of a single type of neural networks, overlooking the challenging scenario of multi-model execution. Hence, the existing methods tend to optimize inference execution for a given  hardware. We are looking for novel solutions that explore the synergies and disparities of multiple inference models acting on a given dataset.

**Testbed and baseline**: There are a few datasets are commonly used in multi-label learning. For instance, Pascal VOC 2007 [^3], MS COCO [^4] and NUS-WIDE [^5]. We encourage using these commonly used datasets as well as the state-of-the-art architectures suited for multi-label learning, since our goal is to design novel algorithms not the testbeds or the architecture. Although each research question intertwines with each other, we expect that each question will be explored first independently and assume the baseline configuration for other research questions. At the last few weeks, we expect to exchange the findings of each question and propose a robust Multi Label Learning algorithm. 

#### Research Question 1: Partial Multi-Label Learning (PML) with deep neural networks by learning the confidence matrix estimation
How to estimate the confidence matrix in Partial Multi-Label Learning with deep neural network? In PML, a subset of labels is relevant to the sample and rest of the labels are irrelevant and each label has a confidence of being ground truth. How to use DNN for confidence matrix estimation for PML?

#### Research Question 2: Online Multi-Label Learning with Crowd
How to train an online multi-label classifier with wrong labels from different crowds? How to aggregate true multi-label from multiple workers (crowd-source) in an online manner?

#### Research Question 3: Multi-Label Classification with Missing Labels with Active Learning
What if in a multi-label classification problem, instead of the whole label set, a subset of labels were provided for each image? Active learning is a method that identifies informative data and uses a human expert for labeling. How can one infer the whole label set per data example while some labels missing, benefiting human knowledge?  

#### Research Question 4: Learning Deep Neural Networks with Multi-Labeled Data with Missing Labels
Given a set of multi-labeled data with missing labels (partially), how could one train a neural network to learn the full label set per example?  

#### Research Question 5: Semi-supervised Multi-Label Classification with Missing Labels
The problem of incomplete labels is frequently encountered in many application domains where the training labels are obtained via crowd-sourcing. Semi-supervised learning is a method that deals with incomplete labels. The question is how one could reconstruct the example-label matrix benefiting semi-supervised learning.

### Prerequisites 
Students shall have basic knowledge of machine learning, deep neural networks and experience in Python and learning framework such as Keras, Tensorflow and Torch. 

### Planning of the research project
1. A kick-off meeting (in Q3)
2. Research proposal presentations (Q4 week 2)
3. Go/no-go presentations (Q4 week 4)
4. Deadline for receiving feedback on final draft (Q4 week 8)



### References

[^1] Min-Ling Zhang and Zhi-Hua Zhou. A review on multi-label learning algorithms.IEEE transac-tions on knowledge and data engineering, 26(8):1819–1837, 2013.

[^2] Gil Levi and Tal Hassner. Age and Gender Classification using Convolutional Neural Networks.2008 8th IEEE International Conference on Automatic Face and Gesture Recognition, FG 2008,2015.

[^3] Mark Everingham, S. M. Ali Eslami, Luc Van Gool, Christopher K. I. Williams, John M. Winn,and Andrew Zisserman.  The pascal visual object classes challenge: A retrospective.Int. J.Comput. Vis., 111(1):98–136, 2015.

[^4] Tsung-Yi Lin, Michael Maire, Serge J. Belongie, James Hays, Pietro Perona, Deva Ramanan,Piotr Dollár, and C. Lawrence Zitnick. Microsoft COCO: common objects in context. In David J.Fleet, Tomás Pajdla, Bernt Schiele, and Tinne Tuytelaars, editors,ECCV, volume 8693 ofLectureNotes in Computer Science, pages 740–755. Springer, 2014.

[^5] Tat-Seng Chua, Jinhui Tang, Richang Hong, Haojie Li, Zhiping Luo, and Yantao Zheng. NUS-WIDE: a real-world web image database from national university of singapore.  In StéphaneMarchand-Maillet and Yiannis Kompatsiaris, editors,CIVR. ACM, 2009. 
