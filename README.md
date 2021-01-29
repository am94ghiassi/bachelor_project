# Robust Multi-Label Learning Deep Neural Networks with Imperfect Labels

## Supervision Team

Responsible Professor: Dr. Lydia Y. Chen

Supervisors: Amirmasoud Ghiassi and Taraneh Younesian

## Background and motivation

Multi-label learning (MLL) is an emerging extension of multi-class classification, which multiple labels are assigned to each data sample. In real-world applications, a data sample usually could be associated with multiple labels. For instance, a landscape picture of Delft in Figure 1, can have the labels "tree",  "bicycle", "canal", and "Delft" at the same time. In MLL, the goal is to train classification models with multi-label example [1] which can predict them with high accuracy. However, acquiring a fully labeled dataset in multi-labeled scenario is a time-consuming and expensive task, and therefore a big challenge in this area is to provide the clean label set. 

The main challenge that we try to address in this project is MLL with all sorts of imperfect labels. This problem can be divided into three sub problems: *i)* MLL with weak labels, *ii)* MLL with wrong labels, and *iii)* MLL with missing labels. 

In many cases usually the provided label set is included with all the relevant labels and partially with some irrelevant labels, which we call MLL with weak labels [2], shown in Figure 1. As seen in the figure, among all the provided labels only a subset of the given labels is true. Then the challenge is to identify the correct labels among the provided label set. A solution for this issue is to assign confidence values to each label per example and learn those values in an optimization problem setting. The prior art [2,3] select high-confidence labels, then use the off-the-shelf MLL methods to train models in the weakly labeled scenario. 

A similar issue arises where not all the relevant labels are provided and instead a few irrelevant labels are, which we call MLL with wrong labels. The challenge here is to identify those wrong labels and replace them with the correct relevant labels. This problem has been addressed in [4] by a dimensionality reduction method based on dependence maximization. 

Moreover, missing labels is another emerging topic in multi-label learning [5]. This issue frequently happen in domains where the labels are collected via crowd-sourcing to reduce the labeling cost and effort. Furthermore, independent from crowd-sourcing, in some applications training based on incomplete label set and predicting the unavailable labels is preferred to reduce the annotation cost. This issue is illustrated in Figure 1, where only some of the true labels are available for the image.

![mml](https://github.com/am94ghiassi/bachelor_project/blob/main/brpimg.png)

Figure 1: An illustration of Multi-Label Learning.


### Research Questions for the Sub-Projects

By the end of this project the question that needs to be answered is *"how to train an accurate multi-labeled classifier with imperfect label information?"*. In the following we bring five research questions which address these emerging challenges in multi-label learning with incomplete information. The end goal is to design algorithms that are robust to incomplete label information and can predict the correct label set per example with high accuracy.

**Expected Novelty** The aforementioned challenges are recently being one of the main focuses of the image classification community. Since real-world images can be associated with more than one label, multi-label learning and its variants are closer to real-world applications. We look for novel solutions and ideas to improve the existing works or addressing their shortcoming.  

**Testbed and baseline**: There are a few datasets are commonly used in multi-label learning. For instance, Pascal VOC 2007 [6], MS COCO [7] and NUS-WIDE [8]. We encourage using these commonly used datasets as well as the state-of-the-art architectures suited for multi-label learning, since our goal is to design novel algorithms not the testbeds or the architecture. Although each research question intertwines with each other, we expect that each question will be explored first independently and assume the baseline configuration for other research questions. At the last few weeks, we expect to exchange the findings of each question and propose a robust Multi Label Learning algorithm. 

#### Research Question 1: Filtering Wrong Labels in Multi-Labeled Data with Weak Labels for Training a Deep Neural Network
Consider the scenario where the multi-labeled data include all relevant labels and some irrelevant labels for each instance, and the irrelevant ones should be filtered out. How to filter wrong labels (irrelevant) and then train DNNs with multi-labeled data which is associated with corrected labels (relevant) via extending the single-label approaches e.g. Co-teaching [9,10]?

#### Research Question 2: Active Learning with Multi-Label Classification with Wrong Labels
How to use human expert to clean/relabel the wrongly labeled multi-label data to obtain high accuracy classification? How could one benefit from active learning to identify informative examples to relabel by the expert?

#### Research Question 3: Wrong Labels Correction in Multi-Labeled Data for Training a Deep Neural Network
Consider the case where the multi-labeled data includes some true and some wrong labels for each instance. The question is how to first identify and correct the wrong labels and then train DNNs with multi-labeled data with corrected labels? One can do this by extending the single-label noise correction methods e.g. D2L [11], GLC [12] and F-correction [13]?

#### Research Question 4: Multi-Label Classification with Missing Labels with Active Learning
What if in a multi-label classification problem, instead of the whole label set, a subset of labels were provided for each image? Active learning is a method that identifies informative data and uses a human expert for labeling. How can one infer the whole label set per data example while some labels missing, benefiting human knowledge?
#### Research Question 5: Training Deep Neural Networks with Multi-Labeled Data with Missing Labels
Given a set of multi-labeled data with missing labels, how could one design a robust loss function for a neural network to learn the full label set per example. Particularly, the network needs to be robust to the missing labels while classifying the test data.

### Prerequisites 
Students shall have basic knowledge of machine learning, deep neural networks and experience in Python and learning framework such as Keras, Tensorflow and Torch. 

### Planning of the research project
1. A kick-off meeting (in Q3)
2. Research proposal presentations (Q4 week 2)
3. Go/no-go presentations (Q4 week 4)
4. Deadline for receiving feedback on final draft (Q4 week 8)



### References

[1] Min-Ling Zhang and Zhi-Hua Zhou. A review on multi-label learning algorithms. IEEE transactions on knowledge and data engineering, 26(8):1819–1837, 2013.

[2] Ming-Kun Xie and Sheng-Jun Huang. Partial multi-label learning. In AAAI, volume 32, 2018.

[3] Gengyu Lyu, Songhe Feng, and Yidong Li. Partial multi-label learning via probabilistic graphmatching mechanism. In SIGKDD, pages 105–113, 2020.

[4] Karl Øyvind Mikalsen, Cristina Soguero-Ruíz, Filippo Maria Bianchi, and Robert Jenssen. Noisy multi-label semi-supervised dimensionality reduction. Pattern Recognit., 90:257–270, 2019.

[5] Thibaut Durand, Nazanin Mehrasa, and Greg Mori.  Learning a deep convnet for multi-label classification with partial labels. In CVPR, pages 647–657. IEEE, 2019.

[6] Mark Everingham, S. M. Ali Eslami, Luc Van Gool, Christopher K. I. Williams, John M. Winn, and Andrew Zisserman. The pascal visual object classes challenge: A retrospective.Int. J.Comput. Vis., 111(1):98–136, 2015.

[7] Tsung-Yi Lin, Michael Maire, Serge J. Belongie, James Hays, Pietro Perona, Deva Ramanan,Piotr Dollár, and C. Lawrence Zitnick. Microsoft COCO: common objects in context. In David J.Fleet, Tomás Pajdla, Bernt Schiele, and Tinne Tuytelaars, editors, ECCV, volume 8693 of Lecture Notes in Computer Science, pages 740–755. Springer, 2014.

[8] Tat-Seng Chua, Jinhui Tang, Richang Hong, Haojie Li, Zhiping Luo, and Yantao Zheng. NUS-WIDE: a real-world web image database from national university of singapore.  In StéphaneMarchand-Maillet and Yiannis Kompatsiaris, editors, CIVR. ACM, 2009.

[9] Bo Han, Quanming Yao, Xingrui Yu, Gang Niu, Miao Xu, Weihua Hu, Ivor Tsang, and Masashi Sugiyama. Co-teaching: Robust training of deep neural networks with extremely noisy labels. In NIPS, pages 8527–8537, 2018.

[10] Xingrui Yu, Bo Han, Jiangchao Yao, Gang Niu, Ivor W Tsang, and Masashi Sugiyama. How does disagreement help generalization against label corruption? In ICML, pages 7164–7173, 2019.

[11] Yisen Wang, Xingjun Ma, Michael E Houle, Shu-Tao Xia, and James Bailey. Dimensionality-driven learning with noisy labels. ICML, pages 3361–3370, 2018.

[12] Dan Hendrycks, Mantas Mazeika, Duncan Wilson, and Kevin Gimpel. Using trusted data to train deep networks on labels corrupted by severe noise. In NIPS, pages 10456–10465, 2018.

[13] Giorgio Patrini, Alessandro Rozza, Aditya Krishna Menon, Richard Nock, and Lizhen Qu. Making deep neural networks robust to label noise: A loss correction approach. In CVPR, pages1944–1952, 2017
