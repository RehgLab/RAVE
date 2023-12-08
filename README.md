### RAVE: Randomized Noise Shuffling for Fast and Consistent Video Editing with Diffusion Models - Official Repo

[Ozgur Kara](https://karaozgur.com/), [Bariscan Kurtkaya](https://bariscankurtkaya.github.io/), [Hidir Yesiltepe](https://sites.google.com/view/hidir-yesiltepe), [James M. Rehg](https://scholar.google.com/citations?hl=en&user=8kA3eDwAAAAJ), [Pinar Yanardag](https://scholar.google.com/citations?user=qzczdd8AAAAJ&hl=en)


<a href='https://arxiv.org/abs/2312.04524'><img src='https://img.shields.io/badge/ArXiv-2312.04524-red'></a> 
<a href='https://rave-video.github.io/'><img src='https://img.shields.io/badge/Project-Page-Green'></a>
[![GitHub](https://img.shields.io/github/stars/rehg-lab/RAVE?style=social)](https://github.com/rehg-lab/RAVE)

![Visitors](https://api.visitorbadge.io/api/visitors?path=https%3A%2F%2Fgithub.com%2Frehg-lab%2FRAVE&label=visitors&countColor=%23263759)


![teaser](assets/examples/grid-2x3.gif)


## Abstract
<b>TL; DR:</b> RAVE is a zero-shot, lightweight, and fast framework for text-guided video editing, supporting videos of any length utilizing text-to-image pretrained diffusion models. 

<details><summary>Click for the full abstract</summary>


> Recent advancements in diffusion-based models have demonstrated significant success in generating images from text. However, video editing models have not yet reached the same level of visual quality and user control. To address this, we introduce RAVE, a zero-shot video editing method that leverages pre-trained text-to-image diffusion models without additional training. RAVE takes an input video and a text prompt to produce high-quality videos while preserving the original motion and semantic structure. It employs a novel noise shuffling strategy, leveraging spatio-temporal interactions between frames, to produce temporally consistent videos faster than existing methods. It is also efficient in terms of memory requirements, allowing it to handle longer videos.  RAVE is capable of a wide range of edits, from local attribute modifications to shape transformations. In order to demonstrate the versatility of RAVE, we create a comprehensive video evaluation dataset ranging from object-focused scenes to complex human activities like dancing and typing, and dynamic scenes featuring swimming fish and boats. Our qualitative and quantitative experiments highlight the effectiveness of RAVE in diverse video editing scenarios compared to existing methods.
</details>

<br>

**Features**:
- *Zero-shot framework*
- *Working fast*
- *No restriction on video length*
- *Standardized dataset for evaluating text-guided video-editing methods*
- *Compatible with off-the-shelf pre-trained approaches (e.g. [CivitAI](https://civitai.com/))*


## Updates
- [12/2023] Paper is available on ArXiv, project webpage is ready and code is released.

### TODO
- [ ] Integrate [FreeU](https://github.com/ChenyangSi/FreeU)
- [ ] Add more examples
- [ ] Share the dataset
- [ ] Prepare a grad.io based GUI
- [X] ~~Integrate MultiControlNet~~
- [X] ~~Adapt CIVIT AI models~~

## Installation and Inference

### Setup Environment
You can use either 'environment.yaml' 
```shell
conda env create -f environment.yml
```
or 'requirements.txt'
```shell
conda create -n rave python=3.8
conda activate rave
conda install pip
pip install -r requirements.txt
```
to set up the Conda environment.

Our code was tested on the following versions:
```shell
timm==0.6.7 torch==2.0.1+cu118 xformers==0.0.20 diffusers==0.18.2 torch.version.cuda==11.8 python==3.8.0
```

### Inference


To run RAVE, please follow these steps:

1- Put the video you want to edit under `data/mp4_videos` as an MP4 file. Note that we suggest using videos with a size of 512x512 or 512x320.

2- Prepare a config file under the `configs` directory. Change the name of the `video_name` parameter to the name of the MP4 file. You can find detailed descriptions of the parameters and example configurations there.

3- Run the following command:
```shell
python scripts/run_experiment.py [PATH OF CONFIG FILE]
```
4- The results will be generated under the `results` directory. Also, the latents and controls are saved under the `generated` directory to speed up the editing with different prompts on the same video.
Note that the names of the preprocessors available can be found in `utils/constants.py`.

### Use Customized Models from CIVIT AI

Our code allows to run any customized model from CIVIT AI. To use these models, please follow the steps:

1- Determine which model you want to use from CIVIT AI, and obtain its index. (e.g. the index for RealisticVision V5.1 is 130072, you can find the id of the model in the website link as a parameter assigned to 'VersionId', e.g. https://civitai.com/models/4201?modelVersionId=130072)

2- In the current directory, run the following code. It downloads the model in safetensors format, and converts it to '.bin' format that is compatible with diffusers.
```shell
bash CIVIT_AI/civit_ai.sh 130072
```
3- Copy the path of the converted model, `$CWD/CIVIT_AI/diffusers_models/[CUSTOMIZED MODEL]` (e.g. `CIVIT_AI/diffusers_models/realisticVisionV60B1_v51VAE` for 130072), and use the path in the config file.


## Dataset

Dataset will be released soon.

## Examples 
### Type of Edits
<table>
<tr>
  <td><img src="assets/examples/glitter.gif"></td>
  <td><img src="assets/examples/watercolor-new.gif"></td>
  <td><img src="assets/examples/coast.gif"></td>
</tr>
<tr>
  <td width=33% style="text-align:center;">1- Local Editing</td>
  <td width=33% style="text-align:center;">2- Visual Style Editing</td>
  <td width=33% style="text-align:center;">3- Background Editing</td>
</tr>
</table>

<table>
<tr>
  <td><img src="assets/examples/a_dinosaur.gif"></td>
  <td><img src="assets/examples/tractor.gif"></td>
</tr>

<tr>
  <td width=50% style="text-align:center;">4- Shape/Attribute Editing</td>
  <td width=50% style="text-align:center;">5- Extreme Shape Editing</td>
</tr>
</table>

### Editing on Various Types of Motions
<table>
<tr>
  <td><img src="assets/examples/crochet.gif"></td>
  <td><img src="assets/examples/anime.gif"></td>
  <td><img src="assets/examples/rave.gif"></td>
</tr>
<tr>
  <td width=33% style="text-align:center;">1- Exo-motion</td>
  <td width=33% style="text-align:center;">2- Ego-motion</td>
  <td width=33% style="text-align:center;">3- Ego-exo motion</td>
</tr>
</table>

<table>
<tr>
  <td><img src="assets/examples/cheetah.gif"></td>
  <td><img src="assets/examples/whales.gif"></td>
</tr>

<tr>
  <td width=50% style="text-align:center;">4- Occlusions</td>
  <td width=50% style="text-align:center;">5- Multiple objects with appearance/disappearance</td>
</tr>
</table>

## Citation 

```
@article{2312.04524,
Author = {Ozgur Kara and Bariscan Kurtkaya and Hidir Yesiltepe and James M. Rehg and Pinar Yanardag},
Title = {RAVE: Randomized Noise Shuffling for Fast and Consistent Video Editing with Diffusion Models},
Year = {2023},
Eprint = {arXiv:2312.04524},
}
``` 

## Maintenance

This is the official repository for **RAVE: Randomized Noise Shuffling for Fast and Consistent Video Editing with Diffusion Models**. Feel free to contact for any questions or discussions [Ozgur Kara](ozgurrkara99@gmail.com).
