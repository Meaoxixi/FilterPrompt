# ___***FilterPrompt: Guiding Image Transfer in Diffusion Models***___

<a href='https://meaoxixi.github.io/FilterPrompt/'><img src='https://img.shields.io/badge/Project-Page-green'></a> 
<a href='https://arxiv.org/pdf/2404.13263'><img src='https://img.shields.io/badge/Paper-blue'></a> 
<a href='https://arxiv.org/pdf/2404.13263'><img src='https://img.shields.io/badge/Demo-orange'></a> 

We propose FilterPrompt, an approach to enhance the model control effect. It can be universally applied to any diffusion model, allowing users to adjust the representation of specific image features in accordance with task requirements, thereby facilitating more precise and controllable generation outcomes. In particular, our designed experiments demonstrate that the FilterPrompt optimizes feature correlation, mitigates content conflicts during the generation process, and enhances the model's control capability.

![arch](https://raw.githubusercontent.com/Meaoxixi/FilterPrompt/gh-pages/resources/method_diagram.png)

---
# Getting Started
## Prerequisites
- Linux or macOS
- NVIDIA GPU (Available memory is greater than 20GB)
- CUDA CuDNN (version ≥ 11.1)
- Python 3.7.16
- PyTorch: [Find the torch version that is suitable for the current cuda](https://pytorch.org/get-started/previous-versions/)
  - 【example】:`pip install torch==1.10.1+cu111 torchvision==0.11.2 torchaudio==0.10.1 -f https://download.pytorch.org/whl/cu111/torch_stable.html`

## Installation
Specifically, inspired by the concept of decoupled cross-attention in [IP-Adapter](https://ip-adapter.github.io/), we apply a similar methodology. 
Please follow the instructions below to complete the environment configuration required for the code:
- Cloning this repo
```
git clone --single-branch --branch main https://github.com/Meaoxixi/FilterPrompt.git
```
- Dependencies

We recommend running this repository using [Anaconda](https://docs.anaconda.com/anaconda/install/). 
All dependencies for defining the environment are provided in `requirements.txt`.
```
cd FilterPrompt
conda create --name fp_env python=3.7.16
conda activate fp_env
pip install torch==1.10.1+cu111 torchvision==0.11.2 torchaudio==0.10.1 -f https://download.pytorch.org/whl/cu111/torch_stable.html
pip install -r requirements.txt
```
- Download the necessary modules in the relative path `models/` from the following links

| Path                                                                                                                 | Description                                                                                                             |
|:---------------------------------------------------------------------------------------------------------------------|:------------------------------------------------------------------------------------------------------------------------|
| `models/`                                                                                                            | root path                                                                                                               |
| &nbsp;&nbsp;&nbsp;&nbsp;├── `ControlNet/`                                                                            | Place the pre-trained model of [ControlNet](https://huggingface.co/lllyasviel)                                          |
| &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;├── `control_v11f1p_sd15_depth `                   | ControlNet_depth                                                                                                        |
| &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;└── `control_v11p_sd15_softedge`                   | ControlNet_softEdge                                                                                                     |
| &nbsp;&nbsp;&nbsp;&nbsp;├── `IP-Adapter/`                                                                            | Place the model of [IP-Adapter](https://huggingface.co/h94/IP-Adapter/tree/main/models)                                 |
| &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;├── `image_encoder `                               | image_encoder of IP-Adapter                                                                                             |
| &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;└── `other needed configuration files`             |                                                                                                                         |
| &nbsp;&nbsp;&nbsp;&nbsp;├── `sd-vae-ft-mse/`                                                                         | Place the model of [sd-vae-ft-mse](https://huggingface.co/stabilityai/sd-vae-ft-mse/tree/main)                          |
| &nbsp;&nbsp;&nbsp;&nbsp;├── `stable-diffusion-v1-5/`                                                                 | Place the model of [stable-diffusion-v1-5](https://huggingface.co/runwayml/stable-diffusion-v1-5)                       |
| &nbsp;&nbsp;&nbsp;&nbsp;├── `Realistic_Vision_V4.0_noVAE/`                                                           | Place the model of [Realistic_Vision_V4.0_noVAE](https://huggingface.co/SG161222/Realistic_Vision_V4.0_noVAE/tree/main) |




## Demo on Gradio

After installation and downloading the models, you can use `python app.py` to perform code in gradio. We have designed four task types to facilitate you to experience the application scenarios of FilterPrompt.

[//]: # (```)

[//]: # (python demo_gradio.py)

[//]: # (```)

[//]: # ()
[//]: # (Note that both images should be size of 1024x1024 to obtain best results.)

[//]: # ()
[//]: # (It should provide the following interface for you to try. Make sure you install DPT following the section above.)

[//]: # ()
[//]: # ()
[//]: # (## Inferencing on batch of images)

[//]: # (To cross-inference on a set of input images and material exemplars, first create the following directory: )

[//]: # ()
[//]: # (```)

[//]: # (mkdir demo_assets/output_images)

[//]: # (```)

[//]: # ()
[//]: # (Follow the above steps to obtain and place all the material exemplars and corresponding input images/depths into their directories.)

[//]: # ()
[//]: # (Then run:)

[//]: # ()
[//]: # (```)

[//]: # (python run_batch.py)

[//]: # (```)

[//]: # ()
[//]: # (### Visualize results using HTML4Vision)

[//]: # ()
[//]: # (To visualize all the batch results, we utilize the HTML4Vision library, which can be installed with:)

[//]: # ()
[//]: # (```)

[//]: # (pip install html4vision)

[//]: # (```)

[//]: # ()
[//]: # (Then, run:)

[//]: # ()
[//]: # (```)

[//]: # (python visualization.py)

[//]: # (```)

[//]: # ()
[//]: # (This will generate an html file `index.html` in the same directory that contains all the results after material transfer.)

## Citation
If you find [FilterPrompt](https://arxiv.org/abs/2404.13263) helpful in your research/applications, please cite using this BibTeX:
```bibtex
@misc{wang2024filterprompt,
      title={FilterPrompt: Guiding Image Transfer in Diffusion Models}, 
      author={Xi Wang and Yichen Peng and Heng Fang and Haoran Xie and Xi Yang and Chuntao Li},
      year={2024},
      eprint={2404.13263},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```
