# BC-MRI-SEG

## A Breast Cancer MRI Tumor Segmentation Benchmark

### Summary
Binary breast cancer tumor segmentation with Magnetic Resonance Imaging (MRI) data is typically trained and evaluated on private medical data, which makes comparing deep learning approaches difficult. We propose a benchmark (BC-MRI-SEG) for binary breast cancer tumor segmentation based on publicly available MRI datasets. The benchmark consists of four datasets in total, where two datasets are used for supervised training and evaluation, and two are used for zero-shot evaluation.

### Benchmark Instructions
1. **Follow the steps in Setup.**
2. **Replicate the Inference Example results.** (Just replicating the RIDER DSC is sufficient)
3. **Use the Training Example as a starting point and apply your own approach.**
4. **Beat the results seen below!** The benchmark expects a model to be trained on 80-20 patient split on the ISPY1 and BreastDM datasets. After it is trained it is then evaluated on test data from ISPY1 and BreastDM and also evaluated on the unseen datasets of RIDER and DUKE. Currently SegResNet achieves the best combined dice score across both evaluations.

![Breast Cancer MRI](images/results.png)

### Setup
1. **Download the four datasets to a single directory named `full_data`.** Download ISPY1 and DUKE with Classic Directory Names. Unzip BreastDM into its own directory named BreastDM.
2. **Rename the top-level directories of each downloaded dataset to ISPY1, RIDER, BreastDM, and DUKE respectively.** Make sure to download DUKE's mapping and boxes files and to place them in the now DUKE directory.
3. **Create a new directory named `data`.**
4. **Run all of the python files for each respective dataset under the `dataprep` directory.**
5. *Optional:* Now all the datasets are ready. If you want to replicate our experiments or run the examples run the `standardize_data.py` file beforehand and download our [weights](https://drive.google.com/file/d/1fcUzheXMvmmrV3CKt0woC_9aHi_ltrwA/view?usp=sharing).

### Dataset Links
- RIDER: [RIDER Breast MRI](https://wiki.cancerimagingarchive.net/display/Public/RIDER+Breast+MRI)
- BreastDM: [Breast-cancer-dataset](https://github.com/smallboy-code/Breast-cancer-dataset)
- ISPY1: [ISPY1 Breast MRI](https://wiki.cancerimagingarchive.net/pages/viewpage.action?pageId=101942541#101942541215b684587f64c8cab1ffc45cd63f339)
- DUKE: [DUKE Breast Cancer MRI](https://www.cancerimagingarchive.net/collection/duke-breast-cancer-mri/)
  - DUKE mapping: File Path mapping tables
  - DUKE box annotations: Annotation Boxes

### Inference Example
To test inference you can run the `test.py` file. If it is running properly you will get a RIDER DSC ~0.13 and a DUKE F1 ~0.27.

### Training Example
To test training you can run the `train.py` file.

### References
Medical-SAM-Adapter (Med-SA)
- [GitHub - Medical-SAM-Adapter](https://github.com/KidsWithTokens/Medical-SAM-Adapter)
- Junde Wu, Wei Ji, Yuanpei Liu, Huazhu Fu, Min Xu, Yanwu Xu, Yueming Jin. "Medical SAM Adapter: Adapting Segment Anything Model for Medical Image Segmentation." arXiv preprint arXiv:2304.12620 (2023). [\[PDF\]](https://arxiv.org/abs/2304.12620)
