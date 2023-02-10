# HEE
This python module offers a series of Spectral Learning techniques for Hypergraph embedding.


## How to use HEE
The implementation of all methods are available in the python script [hee.methods.spectral_methods.py](https://github.com/alessant/HEE/blob/master/hee/methods/spectral_methods.py).

New spectral embedding techniques must be implemented extending the base class `SpectralEmbeddingFramework` and the class must implement the methods: 
- `_laplacian(self)` (private) that returns the hypergraph laplacian matrix;
- `fit(self, dim, **kwargs)` that returns the vertex embeddings matrix.

### Quick look
We provide a [python notebook](https://github.com/alessant/HEE/blob/master/hee/notebook/laplacian_embedding.ipynb) when the implemented techniques are compared on a clustering task on the [ZOO dataset](https://archive.ics.uci.edu/ml/datasets/zoo).

## List of supported methods
- D. Zhou, J. Huang, and B. Schölkopf. 2007. Learning with Hypergraphs: Clustering, Classification, and Embedding. In Proceedings of Neural
Information Processing Systems. 1601–1608
- P. Ren, R. C. Wilson, and E. R. Hancock. 2008. Spectral Embedding of Feature Hypergraphs. In Structural, Syntactic, and Statistical Pattern
Recognition. 308–317
- M. Bolla. 1993. Spectra, Euclidean representations and clusterings of hypergraphs. Discrete Math. 117 (1993), 19–39. Issue 1-
- Y. Zhu, Z. Guan, T. Tan, H. Liu, D. Cai, and X. He. 2016. Heterogeneous hypergraph embedding for document recommendation. Neurocomputing
216 (2016), 150–162
- F. Luo, B. Du, L. Zhang, L. Zhang, and D. Tao. 2019. Feature Learning Using Spatial-Spectral Hypergraph Discriminant Analysis for Hyperspectral
Image. IEEE Trans. on Cybernetics 49, 7 (2019), 2406–2419
- J.A. Rodrìguez. 2002. On the Laplacian Eigenvalues and Metric Parameters of Hypergraphs. Linear and Multilinear Algebra 50, 1 (2002), 1–14.
- S. Saito, D. P. Mandic, and H. Suzuki. 2018. Hypergraph p-Laplacian: A Differential Geometry View. Proceedings of the AAAI Conference on Artificial
Intelligence 32, 1 (2018)
- HGE
- SMHC


## Requirements
- HypernetX
- Numpy
- Scipy
- sklearn
