<h1 id="title">VFX Creator: Animated Visual Effect Generation with Controllable Diffusion Transformer</h1>
<p align="center">
  <!-- <a href="https://liuxinyv.github.io/" target="_blank">Xinyu Liu</a><sup>1</sup>,
  <a href="https://ailingzeng.site/" target="_blank">Ailing Zeng</a><sup>2</sup>, 
  <a href="http://undefined" target="_blank">Wei Xue</a><sup>1</sup>, 
  <a href="http://undefined" target="_blank">Harry Yang</a><sup>1</sup>, 
  <a href="http://undefined" target="_blank">Wenhan Luo</a><sup>1</sup>, 
  <a href="http://undefined" target="_blank">Qifeng Liu</a><sup>1</sup>,
   <a href="http://undefined" target="_blank">Yike Guo</a><sup>1</sup>
  <br><br>
  <sup>1</sup>Hong Kong University of Science and Technology<br>
  <sup>2</sup>Tencent AI Lab<br> -->
</p>
<div align="center">
  <a href="https://vfx-creator0.github.io/"><img src="https://img.shields.io/static/v1?label=Project%20Page&message=Github&color=blue&logo=github-pages"></a> &ensp;
  <a href="https://arxiv.org/abs/2502.05979"><img src="https://img.shields.io/static/v1?label=Paper&message=Arxiv&color=red&logo=arxiv"></a> &ensp;
  <a href="https://huggingface.co/datasets/sophiaa/Open-VFX" target="_blank"><img src="https://img.shields.io/badge/Data-OpenVFX-darkorange" alt="Data"></a>

</div>

<!-- <img src="assets\teaser.jpg" width="800"/> -->

<h2 id="abstract">🔆 Abstract</h2>

<blockquote>
  <p>Crafting magic and illusions stands as one of the most thrilling facets of filmmaking, with visual effects (VFX) serving as the powerhouse behind unforgettable cinematic experiences. While recent advances in generative artificial intelligence have catalyzed progress in generic image and video synthesis, the domain of controllable VFX generation remains comparatively underexplored. More importantly, fine-grained spatial-temporal controllability in VFX generation is critical, but challenging due to data scarcity, complex dynamics, and precision in spatial manipulation. In this work, we propose a novel paradigm for animated VFX generation as image animation, where dynamic effects are generated from user-friendly textual descriptions and static reference images. Our work makes two primary contributions: <strong>i) Open-VFX</strong>, the first high-quality VFX video dataset spanning 15 diverse effect categories, annotated with textual descriptions, instance segmentation masks for spatial conditioning, and start–end timestamps for temporal control; This dataset features a wide range of subjects for the reference images, including characters, animals, products, and scenes. <strong>ii) VFX Creator</strong>, a simple yet effective controllable VFX generation framework based on a Video Diffusion Transformer. The model incorporates a spatial and temporal controllable LoRA adapter, requiring minimal training videos. Specifically, a plug-and-play mask control module enables instance-level spatial manipulation, while tokenized start-end motion timestamps embedded in the diffusion process accompanied by the text encoder, allowing precise temporal control over effect timing and pace.  Extensive experiments on the Open-VFX test set with unseen reference images demonstrate the superiority of the proposed system to generate realistic and dynamic effects, achieving state-of-the-art performance and generalization ability in both spatial and temporal controllability. Furthermore, we introduce a specialized metric to evaluate the precision of temporal control. By bridging traditional VFX techniques with generative techniques, the proposed VFX Creator unlocks new possibilities for efficient, user-friendly, and high-quality video effect generation, making advanced VFX accessible to a broader audience.</p>
</blockquote>

## Note
This repository will be updated soon, including:
- [x] **Arxiv paper**.
- [x] Uploading the **Open-VFX Dataset**.
- [x] Uploading the **Checkpoints** of all visual effects.
- [ ] Uploading the **Annotation Tools**.
- [ ] Uploading the codes of **VFX Creator**.
- [ ] Uploading the **Training** and **Evaluation** scripts.
- [ ] Uploading the **Spatial-temporal Controlled Checkpoints** of visual effects.
      
<h2 id="open-vfx-dataset-overview">🚁 Overview of Open-VFX Dataset</h2>

<p><img src="assets/openvfx.jpg" width="800" alt=""></p>

<h2 id="vfx-creator">😊 VFX Creator</h2>

<div align="center">
  <img src="assets/method.png" width="500" alt="">
</div>

## Installation
Make sure that you have all dependencies in place. The simplest way to do so is to use anaconda.

Make a new conda env and activate it:
```
conda create --name vfxcreator python=3.10
conda activate vfxcreator
```

Install the the packages in the requirements.txt:
```
pip install -r requirements.txt
```
## Checkpoints

We provide all LoRA weights for different VFX:

- [OpenVFX-LoRA-Weights](https://huggingface.co/sophiaa/VFXCreator)


After downloading the weights, you can evaluate the models using the provided script. Please modify the model path and other parameters in the script before running:

```bash
bash test.sh
```

## Training
**For General VFX Generation**:
```bash
bash train.sh
```
Note: Please check the script content before running and modify paths and parameters as needed. For different datasets, please modify the **DATA_ROOT** and **CAPTION_COLUMN**

## Acknowledgment

We would like to thank the open-source data and code of [CogVideo](https://github.com/THUDM/CogVideo), [finetrainers](https://github.com/a-r-r-o-w/finetrainers), [cogvideox-controlnet](https://github.com/TheDenk/cogvideox-controlnet).

## Citation
If you find our work useful in your research, please consider citing:

```bibtex
@article{liu2025vfx,
  title={VFX Creator: Animated Visual Effect Generation with Controllable Diffusion Transformer},
  author={Liu, Xinyu and Zeng, Ailing and Xue, Wei and Yang, Harry and Luo, Wenhan and Liu, Qifeng and Guo, Yike},
  journal={arXiv preprint arXiv:2502.05979},
  year={2025}
}
```

</ul></div></body></html>
