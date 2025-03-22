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
- [ ] Uploading the **Annotation Tools**.
- [ ] Uploading the codes of **VFX Creator**.
- [ ] Uploading the **Training** and **Evaluation** scripts.
- [ ] Uploading the **Checkpoints** of all visual effects.

      
<h2 id="open-vfx-dataset-overview">🚁 Overview of Open-VFX Dataset</h2>

<p><img src="assets/openvfx.jpg" width="800" alt=""></p>

<!-- <h2 id="demonstration">📝 Demonstration</h2>

<table width="1000" align="center" style="margin-top: 30px;">
  <tbody>
    <tr>
      <th style="font-size: 16px">Ta-da it</th>
      <th style="font-size: 16px">Crumble it</th>
      <th style="font-size: 16px">Crush it</th>
      <th style="font-size: 16px">Decapitate it</th>
      <th style="font-size: 16px">Deflate it</th>
    </tr>
    <tr>
      <th><video  class="input_image_comparison" src="assets/Ta-da_it-1.mp4" autoplay loop controls muted></th>
      <th><video  class="input_image_comparison" src="assets/Crumble-2.mp4" autoplay loop controls muted></th>          
      <th><video  class="input_image_comparison" src="assets/Crush-4.mp4" autoplay loop controls muted></th>
      <th><video  class="input_image_comparison" src="assets/Decapitate-2.mp4" autoplay loop controls muted></th>
      <th><video  class="input_image_comparison" src="assets/Deflate-3.mp4" autoplay loop controls muted></th>        </tr>
    <tr>
      <th style="font-size: 16px">Dissolve it</th>
      <th style="font-size: 16px">Explode it</th>
      <th style="font-size: 16px">Eye-pop it</th>
      <th style="font-size: 16px">Inflate it</th>
      <th style="font-size: 16px">Levitate it</th>
    </tr>
    <tr>
      <th><video class="input_image_comparison" src="assets/Dissolve_it-0.mp4" autoplay loop controls muted></th>
      <th><video class="input_image_comparison" src="assets/Explode_it-0.mp4" autoplay loop controls muted></th>
      <th><video  class="input_image_comparison" src="assets/Eye-pop-0.mp4" autoplay loop controls muted></th>
      <th><video  class="input_image_comparison" src="assets/Inflate_it-1.mp4" autoplay loop controls muted></th>
      <th><video class="input_image_comparison" src="assets/Levitate_it-3.mp4" autoplay loop controls muted></th>
    </tr>
    <tr>
      <th style="font-size: 16px">Melt it</th>
      <th style="font-size: 16px">Squish it</th>
      <th style="font-size: 16px">Cake-ify it</th>
      <th style="font-size: 16px">Hot Harley Quinn</th>
      <th style="font-size: 16px">We are Venom</th>
    </tr>
    <tr>
      <th><video class="input_image_comparison" src="assets/Melt_it-3.mp4" autoplay loop controls muted></th>
      <th><video class="input_image_comparison" src="assets/Squish_it-4.mp4" autoplay loop controls muted></th>
      <th><video  class="input_image_comparison" src="assets/Cake-ify-4.mp4" autoplay loop controls muted></th>
      <th><video  class="input_image_comparison" src="assets/harley-1.mp4" autoplay loop controls muted></th>
      <th><video class="input_image_comparison" src="assets/venom-3.mp4" autoplay loop controls muted></th>
    </tr>
    
  </tbody>
</table> -->

<h2 id="vfx-creator">😊 VFX Creator</h2>

<div align="center">
  <img src="assets/method.png" width="500" alt="">
</div>


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
