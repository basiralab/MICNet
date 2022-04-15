# Multigraph-Classification-using-Learnable-Integration-Network

> [Nada Chaari](https://github.com/https://github.com/nadachaari/Multigraph-Classification-using-Learnable-Integration-Network/)<sup>1,2</sup>, [Mohamed Amine Gharsallaoui]<sup>1,3</sup>, [Hatice Camgöz Akdağ]<sup>2</sup>, [Islem Rekik](https://basira-lab.com/)<sup>1</sup>
> <sup>1</sup>BASIRA Lab, Faculty of Computer and Informatics, Istanbul Technical University, Istanbul, Turkey
> <sup>2</sup>Faculty of Management Engineering, Istanbul Technical University, Istanbul, Turkey
> <sup>3</sup>Ecole Polytechnique de Tunisie (EPT), Tunis, Tunisia

> **Abstract:** *Multigraphs with heterogeneous views present one of the most challenging obstacles to classification tasks due to their complexity. Several works based on feature selection have been recently proposed to disentangle the problem of multigraph heterogeneity. However, such techniques have major drawbacks. First , the bulk of such works lies in the vectorization and the flattening operations, failing to preserve and exploit the rich topological properties of the multigraph. Second , they learn the classification process in a dichotomized manner where the cascaded learning steps are pieced in together independently. Hence, such architectures are inherently agnostic to the cumulative estimation error from step to step. To overcome these drawbacks, we introduce MICNet (multigraph integration and classifier network), the first end-to-end graph neural network based model for multigraph classification. First, we learn a single-view graph representation of a heterogeneous multigraph using a GNN based integration model. The integration process in our model helps tease apart the heterogeneity across the different views of the multigraph by generating a subject-specific graph template while pre- serving its geometrical and topological properties. Second, we classify each integrated template using a geometric deep learning block which enables
us to grasp the salient graph features. We train, in end-to-end fashion, these two blocks using a single objective function to optimize the classification performance. We evaluate our MICNet in gender classification using brain multigraphs derived from different cortical measures. We demonstrate that our MICNet significantly outperformed its variants thereby showing its great potential in multigraph classification.*


# Detailed proposed framework pipeline

This work has been accepted in the Journal of Neural Networks, 2022. Our framework, named multigraph integration
and classi er networks (MICNet), is the first graph neural network model that integrates and classifies multigraphs in an end-to-end fashion based on geometric deep learning architecture. Our learning-based framework comprises two key steps. (1) Learning to optimally construct single-view graphs
from the original heterogeneous multigraphs (Integration block), (2) Embedding the nodes across the layers using consecutive GNN-based architecture to predict the target (Classification block). Experimental results against comparison methods demonstrate that our framework can achieve the best results in terms of classification accuracy. We evaluated our proposed framework from brain genomics superstruct project datasets (https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/25833).

More details can be found at: (link to the paper) (https://www.researchgate.net/publication/359539534_Multigraph_Classification_using_Learnable_Integration_Network_with_Application_to_Gender_Fingerprinting).
<p align="center">
  <img src="./MICNet architecture.png">
</p>

# Libraries to preinstall in Python
* [Python 3.8](https://www.python.org/)
* [PyTorch 1.4.0](http://pytorch.org/)
* [Torch-geometric](https://github.com/rusty1s/pytorch_geometric)
* [seaborn 0.11.2](https://pypi.org/project/seaborn/)
* [sklearn 0.0](https://pypi.org/project/sklearn/)
* [Matplotlib 3.1.3+](https://matplotlib.org/)
* [Numpy 1.18.1+](https://numpy.org/)


# Data format
We run a connectomic dataset which consists of 308 human male subjects (M) and 391 human male subjects (F) from the Brain Genomics Superstruct Project (GSP) dataset (Holmes et al., [2015] ( https://www.nature.com/articles/sdata201531/fig_tab )). Each subject is represented by 4 cortical morphological brain networks derived from maximum principal curvature, mean cortical thickness, mean sulcal depth, and average curvature measurements. For each hemisphere, the cortical surface is reconstructed from T1-weighted MRI using FreeSurfer pipeline (Fischl et al., [2012] ( https://pubmed.ncbi.nlm.nih.gov/22248573/ ))and parcellated into 35 cortical regions of interest (ROIs) using Desikan-Killiany cortical atlas (Desikan et al., [2006] ( https://www.sciencedirect.com/science/article/pii/S1053811906000437?casa_token=_b-7Vr1dT9gAAAAA:5OoJBGcMy2AUtVzRGlHtxggoUjIwiMA5H_UFxxiV0ST4WckwB5Zbnv7RwYyO5INXqYnJ-v2S )). The corresponding connectivity strength between two ROIs is derived by computing the absolute difference in their average cortical attribute (e.g., thickness).

Each subject of multi-view brain networks dataset can be represented as a stacked adjacency matrices with shape
```
[num_ROIs x num_ROIs x num_Views]
```
where `num_ROIs` is number of region of interests in the brain graph and `num_views` is number of cortical morphological brain networks, called also views (eg.cortical thickness).
```
[num_Subs x num_ROIs x num_ROIs x num_Views]
```
where num_sub is number of subjects in a dataset. 


**Train and test MICNet**

To evaluate our framework, we used 5-folds cross validation strategy. We evaluate 10 models constructed from the 10 possible combinations of the
integration methods (MGI (Our integartion block), Linear, Average, SNF and netNorm) and classifiers (DIFF and GCN). Based on the accuracy, our MICNet
outperforms all benchmark methods by achieving the highest accuracy rate for subpopulations (5 folds). On the other hand, if we replace the classi er with GCN and combine it with our integration, the resulting model outperforms all the other GCN-based models.

# Python Code
To run MICNet, first, generate a multi-view dataset with dimension shape `[num_Subs x num_ROIs x num_ROIs x num_Views]`. Next, use k-folds cross-validation to divide each dataset into training dataset and testing dataset. Then, train and test MICNet using the code named 'main_MGI_DIFF' above. To benchmark with the other models, use the codes entiteled 'main_netNorm_DIFF', 'main_SNF_DIFF' , etc...

# Example Result

* The figure below demonstrates the classification accuracy distribution by our proposed method (MICNet) and benchmark methods for gender classi cation using right hemisphere cortical multigraphs. Our model MICNet is represented by our integration and our classifier (DIFF) networks. The benchmark methods used for comparison include different integration techniques and classifiers. The integration methods are: similarity network fusion technique (SNF) (Wang et al., 2014), normalization method (netNorm) (Dhifallah et al., 2020), simple average (average), weighted linear average (linear) and our deep integration network. The classifiers are graph convolutional networks (GCN) (Kipf and Welling, 2016) and the classi cation block (DIFF) that we integrated in our model.

<p align="center">
  <img src="./Classification accuracy.png">
</p>

# Please Cite the Following paper when using MICNet:


```latex
@article{chaari2022multigraph,
  title={Multigraph classification using learnable integration network with application to gender fingerprinting},
  author={Chaari, Nada and Gharsallaoui, Mohammed Amine and Akda{\u{g}}, Hatice Camg{\"o}z and Rekik, Islem},
  journal={Neural Networks},
  year={2022},
  publisher={Elsevier}
}
```

https://www.sciencedirect.com/science/article/abs/pii/S0893608022001137
