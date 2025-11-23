### PyTorch implementation of MHRN for benchmark datasets
The proposed methods were implemented using software library PyTorch on a PC server equipped with
NVIDIA TITAN V and Intel i9 10900K.


### Running a model:

     FB15k-237: run main.py (Notes: --algorithm MHRN --default dataset FB15k-237)


### Hyperparameters: 

Available in hyperparameters.txt 


### Requirements

	PyTorch	2.0.0
	Python 3.8
	GrouPy

GrouPy is the implementation of group equivariant convolution. We have already installed GrouPy in our code.
Access GrouPy (Pytorch) at https://github.com/adambielski/GrouPy 

Additionally, we have already downloaded datasets in data folder.
Access FB-IMG, WN9-IMG, and their pretrained embeddings at https://drive.google.com/file/d/1TV92angj47IHC666GBTxhJoEtIQKUZ-N/view?usp=sharing
	
### Citation:
Please cite the following paper if you use this code in your work.

```bibtex
@article{MHRN,
author = {Zengcan Xue and Zhaoli Zhang and Hai Liu and Zhifei Li and Shuyun Han and Erqi Zhang},
title = {MHRN: A multi-perspective hierarchical relation network for knowledge graph embedding},
journal = {Knowledge-Based Systems},
volume = {313},
pages = {113040},
year = {2025},
issn = {0950-7051},
doi = {https://doi.org/10.1016/j.knosys.2025.113040},
url = {https://www.sciencedirect.com/science/article/pii/S0950705125000875},
}
```

### Acknowledgment: 
For any clarification, comments, or suggestions please create an issue or contact xuezc@mails.ccnu.edu.cn (https://zengcanxue.github.io/).
