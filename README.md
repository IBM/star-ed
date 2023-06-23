# Fairness in Dermatology

Images depicting dark skin tones are significantly under-represented in the educational materials used to teach primary care physicians and dermatologists to recognize skin diseases. This could contribute to disparities in skin disease diagnosis across different racial groups. Previously, domain experts have manually assessed textbooks to estimate the diversity in skin images. Manual assessment does not scale to many educational materials and introduces human errors. To automate this process, we present the Skin Tone Analysis for Representation in EDucational materials (STAR-ED) framework, which assesses skin tone representation in medical education materials using machine learning. Given a document (e.g., a textbook in .pdf), STAR-ED applies content parsing to extract text, images, and table entities in a structured format. Next, it identifies images containing skin, segments the skin-containing portions of those images, and estimates the skin tone using machine learning. STAR-ED was developed using the Fitzpatrick17k dataset. We then externally tested STAR-ED on four medical textbooks. Results show strong performance in detecting skin images (0.96±0.02 AUROC and 0.90±0.06 F1 score) and classifying skin tones (0.87±0.01 AUROC and 0.91±0.00 F1 score). STAR-ED quantifies the imbalanced representation of skin tones in four medical textbooks: brown and black skin tones (Fitzpatrick V-VI) images constitute only 10.5% of all skin images. We envision this technology as a tool for medical educators, publishers, and practitioners to assess skin tone diversity in their educational materials. 

![Overview](/figures/approach_fair_derma.drawio.png)

This repo contains code from the paper: **Skin Tone Analysis for Representation in Educational Materials (STAR-ED) Using Machine Learning.** *Girmaw Abebe Tadesse, Celia Cintas, Kush R. Varshney, Peter Staar, Chinyere Agunwa, Skyler Speakman, Justin Jia, Elizabeth E Bailey, Ademide Adelekun, Jules B. Lipoff, Ginikanwa Onyekaba, Jenna C. Lester, Veronica Rotemberg, James Zou and Roxana Daneshjou.* 

## Repo organization

- See *this notebook (TBD)* to see python files to extract images from dermatology textbooks using the structured file (JSON) of the ingestion output using [Deep Search](https://ds4sd.github.io/) and to run a run binary classification of skin vs non-skin images (block **b** and **c** in Figure).
- See [this notebook](https://github.ibm.com/Celia-Cintas/fairderma-public/blob/main/segmentation_and_skintone_classification.ipynb) to see experiments in skin segmentation, feature extraction (e.g ITA values), and skin tone models with traditional machine learning as well as deep learning (block **d** and **e** in Figure).

## Citation

If you find useful this repo please consider to cite:
TBD

## Setup 

`$ python -m venv testvenv`

`$ source testvenv/bin/activate`

`$ pip install -r requirements.txt`

`$ ipython kernel install --user --name=testvenv`

`$ python -m notebook`
