# data-emphasized-ELBO
[Learning Hyperparameters via a Data-Emphasized Variational Objective](https://arxiv.org/abs/2502.01861) by Ethan Harvey, Mikhail Petrov, and Michael C. Hughes

![Figure 2](./notebooks/computational_time_comparison.png)
Figure 3: Test accuracy over time for L2-SP transfer learning methods. We run each method on 3 separate train sets of size $N$ (3 marker styles). Each panel shows a distinct task: ConvNeXt-Tiny fine-tuned on CIFAR-10, Flower-102, and Pet-37 and BERT-base fine-tuned on News-4. We compare MAP + GS, MAP + BO, diagEF LA-LML (Immer et al., 2021), diagEF LA-CLML (Lotfi et al., 2022), iso ELBO, and iso DE-ELBO (ours). **Takeaway: After just a few hours, iso DE-ELBO reaches as good or better performance at small data sizes and similar performance at large sizes, even when other methods are given many additional hours.**

## Citation
```bibtex
@article{harvey2025learning,
  author={Harvey, Ethan and Petrov, Mikhail and Hughes, Michael C},
  title={Learning Hyperparameters via a Data-Emphasized Variational Objective},
  journal={arXiv preprint arXiv:2502.01861},
  year={2025},
}
```
```bibtex
@inproceedings{harvey2024learning,
  author={Ethan Harvey and Mikhail Petrov and Michael C Hughes},
  title={Learning the Regularization Strength for Deep Fine-Tuning via a Data-Emphasized Variational Objective},
  booktitle={NeurIPS 2024 Workshop on Fine-Tuning in Modern Machine Learning: Principles and Scalability},
  year={2024},
  url={https://openreview.net/forum?id=wzvP0CJ8h4},
}
```
