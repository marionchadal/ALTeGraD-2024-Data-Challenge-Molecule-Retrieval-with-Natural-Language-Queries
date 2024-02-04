# ALTeGraD-2024-Data-Challenge-Molecule-Retrieval-with-Natural-Language-Queries

The aim of the challenge is to map molecule graphs with their plain text properties description. We work with a method called contrastive learning, which consists in embedding both structured data in a way that the matching inputs are close one from another in the embedding space. Over the training, the embedding weights are modified in order to pull the matching data embedding and push the different elements apart. Then by evaluating the distance between all the embedding we can deduce a solution to the matching problem.

To handle this challenge, we chose Graph Isomorphism Network (GIN) for our graph encoder due to its foundational principle, which leverages the Weisfeiler-Lehman (WL) graph isomorphism test. The WL test is a classical algorithm for graph isomorphism that iteratively aggregates and hashes neighboring node labels to identify unique structures within graphs. This process effectively captures the topological information of graphs, allowing for a distinction between non-isomorphic structures. GIN, proposed by Xu et al., extends this concept by learning an injective function to aggregate neighborhood features. This approach ensures that the representation power of GIN is at least as powerful as the WL test in distinguishing graph structures, making it highly effective for encoding molecular graphs where the precise arrangement of atoms and bonds is crucial.

# References
Aaron van den Oord, Yazhe Li, and Oriol Vinyals. “Representation Learning with Con-
trastive Predictive Coding”. In: arXiv:1807.03748 (2018)

Jacob Devlin et al. “Bert: Pre-training of deep bidirectional transformers for language
understanding”. In: arXiv preprint arXiv:1810.04805 (2018)

Keyulu Xu et al. “How powerful are graph neural networks?” In: arXiv preprint arXiv:1810.00826
(2018)
