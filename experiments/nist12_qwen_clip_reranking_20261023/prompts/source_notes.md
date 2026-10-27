# RoboGround attribute prompts

Source: Haifeng Huang, Xinyi Chen, Yilun Chen, Hao Li, Xiaoshen Han, Zehan
Wang, Tai Wang, Jiangmiao Pang and Zhou Zhao, *RoboGround: Robotic
Manipulation with Grounded Vision-Language Priors*, CVPR 2025,
[arXiv:2504.21530v1](https://arxiv.org/abs/2504.21530v1).

The exact published prompt wording is transcribed below from supplementary
[Figure 6, PDF page 14](https://arxiv.org/pdf/2504.21530v1#page=14) and
[Figure 7, PDF page 15](https://arxiv.org/pdf/2504.21530v1#page=15).
PDF line wrapping is normalized to paragraphs; spelling, examples, punctuation,
tags and wording are retained. The article's arXiv record links to
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/), which permits attributed
copying and adaptation. No endorsement by the authors is implied.

## Trial adaptation

The published opening is `This image shows four different views of a single banana.`
The trial changes only that sentence to `This image shows a single object.`
in both prompt files, used identically for CAD renders and scene crops. Thus the
unknown crop receives no ground-truth class name, and the single-image request
does not falsely claim four views. No additional system instruction is supplied.
No source instructions or examples are removed. There are two independent
requests per image: keywords and descriptive phrases. The tags are COLOR, SHAPE
and MATERIAL; the published prompts do not request a class tag.

This deliberately retains the source's color/material fields and instruction to
ignore minor object parts. A separate shape-field scoring control can test the
earlier geometry-only preference without rewriting model outputs or these prompts.
The prompts do not impose this project's protected five-word/no-count CAD prompt.
`docs/cad_view_prompt.txt` remains untouched.

The paper's Section 3.2 uses a four-view composite per known object and uses CLIP
encodings for distractor sampling. It does not describe training CLIP on the
generated image-description pairs. This trial uses Qwen instead of GPT-4 and
frozen pretrained CLIP for CAD-to-crop reranking; it is an adapted classification
experiment, not a reproduction of RoboGround's runtime grounding policy.

## Figure 6: published keyword prompt

```text
This image shows four different views of a single banana. Please describe its attributes following these rules:
1. The format should be: <COLOR> its color </COLOR> <SHAPE> its shape </SHAPE> <MATERIAL> its material </MATERIAL>.
2. For each attribute, only include the descriptive words. For example, if you describe the color as "yellow surface with brown spots", please simplify it to "yellow, brown".
3. Do not describe minor parts of the object. For example, if the picture shows a red apple with a small green stem, the color description should be "red" rather than "red, green".
Here are three example outputs for your reference:
1. <COLOR> red, yellow </COLOR> <SHAPE> round </SHAPE> <MATERIAL> organic </MATERIAL>.
2. <COLOR> silver </COLOR> <SHAPE> elongated, round </SHAPE> <MATERIAL> metal, plastic </MATERIAL>.
3. <COLOR> blue </COLOR> <SHAPE> cylindrical </SHAPE> <MATERIAL> aluminum </MATERIAL>.
Now please describe the object shown in the image.
```

## Figure 7: published descriptive-phrase prompt

```text
This image shows four different views of a single banana. Please describe its attributes following the format: <COLOR> its color </COLOR> <SHAPE> its shape </SHAPE> <MATERIAL> its material </MATERIAL>.
Here are four example outputs for your reference:
1. <COLOR> green on top and yellow on the bottom </COLOR> <SHAPE> rectangular prism </SHAPE> <MATERIAL> sponge </MATERIAL>.
2. <COLOR> white </COLOR> <SHAPE> cylindrical with a flip-top lid </SHAPE> <MATERIAL> plastic </MATERIAL>.
3. <COLOR> blue </COLOR> <SHAPE> flat with an elongated handle and slotted flipper </SHAPE> <MATERIAL> plastic or metal </MATERIAL>.
4. <COLOR> brown cap and white stem </COLOR> <SHAPE> typical mushroom shape with a rounded cap and cylindrical stem </SHAPE> <MATERIAL> organic fungus </MATERIAL>.
Now please describe the object shown in the image.
```
