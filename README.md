# Barbie: Text to Barbie-Style 3D Avatars

[**Project Page**](https://xiaokunsun.github.io/Barbie.github.io) | [**Arxiv**](https://arxiv.org/pdf/2408.09126) | [**Gallery**](https://drive.google.com/drive/folders/1FXDROWXrnsSQiOZ4vBgA_Yzib3irLNBc?usp=sharing)

Official repo of "Barbie: Text to Barbie-Style 3D Avatars"

[Xiaokun Sun](https://xiaokunsun.github.io), [Zhenyu Zhang](https://jessezhang92.github.io), [Ying Tai](https://tyshiwo.github.io/index.html), [Hao Tang](https://ha0tang.github.io), [Zili Yi](https://zili-yi.github.io), [Jian Yang](https://scholar.google.com.hk/citations?user=6CIDtZQAAAAJ), 


<p align="center"> All Code will be released soon... 🚀🚀🚀 </p>

Abstract: To integrate digital humans into everyday life, there is a strong demand for generating high-quality, fine-grained disentangled 3D avatars that support expressive animation and simulation capabilities, ideally from low-cost textual inputs. Although text-driven 3D avatar generation has made significant progress by leveraging 2D generative priors, existing methods still struggle to fulfill all these requirements simultaneously. To address this challenge, we propose Barbie, a novel text-driven framework for generating animatable 3D avatars with separable shoes, accessories, and simulation-ready garments, truly capturing the iconic ``Barbie doll'' aesthetic. The core of our framework lies in an expressive 3D representation combined with appropriate modeling constraints. Unlike previous methods, we innovatively employ G-Shell to uniformly model both watertight components (e.g., bodies, shoes, and accessories) and non-watertight garments compatible with simulation. Furthermore, we introduce a well-designed initialization and a hole regularization loss to ensure clean open surface modeling. These disentangled 3D representations are then optimized by specialized expert diffusion models tailored to each domain, ensuring high-fidelity outputs. To mitigate geometric artifacts and texture conflicts when combining different expert models, we further propose several effective geometric losses and strategies. Extensive experiments demonstrate that Barbie outperforms existing methods in both dressed human and outfit generation. Our framework further enables diverse applications, including apparel combination, editing, expressive animation, and physical simulation.

<p align="center">
    <img src="assets/teaser.png">
</p>


## BibTeX

```bibtex
@article{sun2024barbie,
  title={Barbie: Text to Barbie-Style 3D Avatars},
  author={Sun, Xiaokun and Zhang, Zhenyu and Tai, Ying and Tang, Hao and Yi, Zili and Yang, Jian},
  journal={arXiv preprint arXiv:2408.09126},
  year={2024}
}
```
