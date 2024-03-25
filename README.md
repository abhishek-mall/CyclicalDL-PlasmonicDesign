# A Cyclical Deep Learning Based Framework For Simultaneous Inverse and Forward design of Nanophotonic Metasurfaces

## Description
The conventional approach to nanophotonic metasurface design and optimization for a targeted electromagnetic response involves exploring large geometry and material spaces. This is a highly iterative process based on trial and error, which is computationally costly and time consuming. Moreover, the non-uniqueness of structural designs and high non-linearity between electromagnetic response and design makes this problem challenging. To model this unintuitive relationship between electromagnetic response and metasurface structural design as a probability distribution in the design space, we introduce a framework for inverse design of nanophotonic metasurfaces based on cyclical deep learning (DL). The proposed framework performs inverse design and optimization mechanism for the generation of meta-atoms and meta-molecules as metasurface units based on DL models and genetic algorithm. The framework includes consecutive DL models that emulate both numerical electromagnetic simulation and iterative processes of optimization, and generate optimized structural designs while simultaneously performing forward and inverse design tasks. A selection and evaluation of generated structural designs is performed by the genetic algorithm to construct a desired optical response and design space that mimics real world responses. Importantly, our cyclical generation framework also explores the space of new metasurface topologies. As an example application of the utility of our proposed architecture, we demonstrate the inverse design of gap-plasmon based half-wave plate metasurface for user-defined optical response. Our proposed technique can be easily generalized for designing nanophtonic metasurfaces for a wide range of targeted optical response.

## Code
This repository contains the template code used in the research article. The code demonstrates the implementation of a generative adversarial network(GAN), a feed forward network (SNN) and pseudo-genetic algorithm (GA) for the design of plasmonic metasurfaces. The dataset was generated using simulation software, COMSOL.

The program is pure-Python and runs on GPU. Additionally, the following libraries are used:

- numpy
- pandas
- pytorch

## Article Information

[Link to Research Article](https://www.nature.com/articles/s41598-020-76400-y)

- **Authors:** Abhishek Mall, Abhijeet Patil, Amit Sethi, Anshuman Kumar
- **Publication Date:** 2020/11/10
- **Journal:** Scientific Reports
- **Volume:** 1
- **Issue:** 10
- **Pages:** 19427
- **Publisher:** Springer Science and Business Media {LLC}


## Citation
If you use this code or the research in your work, please cite the following:

```plaintext

@article{mall2020cyclical,
  title={A cyclical deep learning based framework for simultaneous inverse and forward design of nanophotonic metasurfaces},
  author={Mall, Abhishek and Patil, Abhijeet and Sethi, Amit and Kumar, Anshuman},
  journal={Scientific reports},
  volume={10},
  number={1},
  pages={19427},
  year={2020},
  publisher={Nature Publishing Group UK London}
}

