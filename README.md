### Setup

1. Download the four datasets to a single directory named `full_data`. Download ISPY1 and DUKE with Classic Directory Names. Unzip BreastDM into its own directory named BreastDM.
2. Rename the top-level directories of each downloaded dataset to ISPY1, RIDER, BreastDM, and DUKE respectively. Make sure to download DUKE's mapping and boxes files and to place them in the now DUKE directory.
3. Create a new directory named `data`.
4. Run all of the python files for each respective dataset under the `dataprep` directory.
5. Optional: Now all the datasets are ready. If you want to replicate our experiments or run the examples run the `standardize_data.py` file beforehand.

### Dataset Links

- RIDER: [RIDER Breast MRI](https://wiki.cancerimagingarchive.net/display/Public/RIDER+Breast+MRI)
- BreastDM: [Breast-cancer-dataset](https://github.com/smallboy-code/Breast-cancer-dataset)
- ISPY1: [ISPY1 Breast MRI](https://wiki.cancerimagingarchive.net/pages/viewpage.action?pageId=101942541#101942541215b684587f64c8cab1ffc45cd63f339)
- DUKE: [DUKE Breast Cancer MRI](https://www.cancerimagingarchive.net/collection/duke-breast-cancer-mri/)
- DUKE mapping: File Path mapping tables
- DUKE box annotations: Annotation Boxes

### Inference Example

To test inference you can run the `test.py` file.

### Training Example

To test training you can run the `train.py` file.

### References

Medical-SAM-Adapter (Med-SA)
- [GitHub - Medical-SAM-Adapter](https://github.com/KidsWithTokens/Medical-SAM-Adapter)
