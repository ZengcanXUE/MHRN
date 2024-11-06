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
Access FB-IMG, WN9-IMG, and their pretrained embeddings at https://github.com/UKPLab/starsem18-multimodalKB https://drive.google.com/file/d/1TV92angj47IHC666GBTxhJoEtIQKUZ-N/view?usp=sharing
	

### Acknowledgment: 
For any clarification, comments, or suggestions please create an issue or contact xuezc@mails.ccnu.edu.cn.
