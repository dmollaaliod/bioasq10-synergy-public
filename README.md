# bioasq10-synergy-public
Public code for Macquarie University's contribution to BioASQ 10 Synergy.

If you use this code, please cite the following paper:

D. Mollá (2022). Query-Focused Extractive Summarisation for Biomedical and COVID-19 Question Answering. *CLEF2022 Working Notes*.

## Files needed

Besides the files available in this repository, you need to obtain the following file. It is not in the github repository because of the large size:

- task10b_distilbert_model_32.pt (254MB) (please contact diego.molla-aliod@mq.edu.au)

## To run the system

The following code replicates the results of runs MQ1 and MQ2 at round 4.

```
$ conda env create -f environment.yml
$ conda activate bioasq10-synergy
$ conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch
(bioasq8-synergy) $ python run_mq1.py
(bioasq8-synergy) $ python run_mq2.py
```
