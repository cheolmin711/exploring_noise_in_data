## Exploring Noise in Data: Applications to ML Models

Website for an introduction to our project:
https://cheolmin711.github.io/exploring_noise_in_data/

Our experiment is based on the foundings of these papers:
Belkin, M., Hsu, D., Ma, S., & Mandal, S. (2019). Reconciling modern machine-learning practice and the classical bias–variance trade-off. Proceedings of the National Academy of Sciences, 116(32), 15849–15854. https://doi.org/10.1073/pnas.1903070116
Belkin, M., Ma, S., & Mandal, S. (2018). To understand deep learning we need to understand kernel learning. arXiv.org. https://arxiv.org/abs/1802.01396

### Models Supported:
Kernel machines, Random Forests, k-Nearest Neighbor Classification

### Building the Project:
Please build the project using the Docker container located at the DockerHub repo in submission.json

### Running the Project:
To run on all data:
> python3 run.py

To run on all data with a custom config file:
> python3 run.py all [json config file]

To run the code on test section of data:
> python3 run.py test

To clean all output files:
> python3 run.py clean

