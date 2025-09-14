# MFANet
code for paper: ['Aligning First, Then Fusing: A Novel Lightweight Multimodal Feature Alignment Network For Short Video Fake News Detection']
## Environment
please refer to the file requirements.txt.
## Dataset
We conduct experiments on two datasets: FakeSV and FakeTT. 
### FakeSV
[FakeSV](https://github.com/ICTMCG/FakeSV) is the largest publicly available Chinese dataset for fake news detection on short video platforms, featuring samples from Douyin and Kuaishou. 
### FakeTT
[FakeTT](https://github.com/ICTMCG/FakingRecipe?tab=readme-ov-file) collect news videos from the TikTok platform, following a similar collection process as FakeSV, provides video, audio and textual descriptions.
## Data Preprocess
- For FakeTT dataset, we use [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR) to extract OCR.
- Pretrained bert-wwm can be downloaded [here](https://drive.google.com/file/d/1-2vEZfIFCdM1-vJ3GD6DlSyKT4eVXMKq/view), and the folder is already prepared in the project.
- We extract features from the original videos using pre-trained [MAE](https://github.com/facebookresearch/mae) and [Hubert](https://github.com/bshall/hubert).
## Train
After placing the data, start training the model:
```python
python main.py
```
