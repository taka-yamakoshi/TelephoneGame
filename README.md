# Probing BERT's priors with serial reproduction chains (Findings of ACL)

<img src="https://user-images.githubusercontent.com/60325582/159703554-90b91ef7-57cd-4d2d-a1f1-f825285a028d.png" width="300">

* [Paper](https://aclanthology.org/2022.findings-acl.314/)
* [Interactive demo](https://huggingface.co/spaces/taka-yamakoshi/bert-priors-demo)

## Repository organization

* `model` contains code to run chains sampling sentences from BERT.

* `corpus` contains code for extracting comparable sentences from wikipedia and bookcorpus.

* `analysis` contains code for analyzing behavioral experiment data, performing corpus comparison, and producing figures.
