# Source credits

- HERO supplies the ViMo motion decoder, normalization statistics, and the video backbone checkpoint. The included inference adaptations derive from its public implementation. [HERO](https://github.com/JackYu6/HERO_release) · [MIT notice](licenses/HERO.txt)
- The vision transformer adapts TC-CLIP from NAVER Cloud Corp. to return RGB context and per-frame features. The shared weight file includes the frozen image encoder from the HERO distribution. [TC-CLIP](https://github.com/naver-ai/tc-clip) · [CC BY-NC 4.0](licenses/TC-CLIP.txt)
- Token merging in the visual encoder uses code from Meta Platforms, Inc. and affiliates. [ToMe](https://github.com/facebookresearch/ToMe) · [CC BY-NC 4.0](licenses/ToMe.txt)
- The visual layer normalization and QuickGELU activation derive from OpenAI CLIP. [CLIP](https://github.com/openai/CLIP) · [MIT notice](licenses/CLIP.txt)
- Joint recovery uses quaternion operations adapted from QuaterNet, copyright Facebook, Inc. [QuaterNet](https://github.com/facebookresearch/QuaterNet) · [CC BY-NC 4.0](licenses/QuaterNet.txt)
- The sampler adapts Wenliang Zhao's UniPC data-prediction update to motion latents and applies adaptive projected guidance. [UniPC](https://github.com/wl-zhao/UniPC) · [MIT notice](licenses/UniPC.txt) · [Adaptive Projected Guidance](https://arxiv.org/abs/2410.02416)
