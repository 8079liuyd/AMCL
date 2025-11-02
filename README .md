# Attention-Guided Multi-View Contrastive Learning for Predicting Sparse  Drug–Gene Associations

**Model is AMCL！**

Accelerating drug-gene interaction prediction is crucial using deep learning methods for drug discovery and repurpos-  
ing. However, the inherent sparsity of data verified by experiments constitutes a major challenge, which often limits  
the robustness and generalization ability of existing prediction models. Therefore, to solve this problem, we propose  
an attention-guided multi-view contrastive learning approach to predicting unknown drug-gene associations, named  
AMCL.Specifically, AMCL integrates multi-scale feature learning, captures local topological information by graph  
convolution network, and extracts global structural patterns by kernel function. The dynamic hypergraph learning  
module is utilized to model high-order dependencies dynamically.The key is the attention bias mechanism based on  
local connectivity aggregate, which guides the model to prioritize the information of densely connected areas in the  
interactive network and assists the prediction task. The cross-view contrastive learning strategy enhances the discrim-  
ination ability of learning embedding, especially under the condition of sparse data. A large number of experiments  
on DGIdb 5.0 data sets show that AMCL is significantly superior to the seven most advanced methods at present. Ab-  
lation studies have verified the key contributions of each component, and case studies have pointed out the versatility  
of AMCL in discovering new drugs and repositioning drugs.

#  Requirements

-   Python 3.10.18
-   pandas  2.3.3
-   numpy                    2.2.6
-   torch                    2.8.0
-   tqdm                     4.67.1
-   scikit-learn             1.7.2


#   Run the demo

```
python main.py
```


