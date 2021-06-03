# Adversarial learning of cancer tissue representations
 
**Abstract:**

*Deep learning based analysis of histopathology images shows promise in advancing understanding of tumor progression, tumor micro-environment, and their underpinning biological processes. So far, these approaches have focused on extracting information associated with annotations. In this work, we ask how much information can be learned from the tissue architecture itself.*

*We present an adversarial learning model to extract feature representations of cancer tissue, without the need for manual annotations. We show that these representations are able to identify a variety of morphological characteristics across three cancer types: Breast, colon, and lung. This is supported by 1) the separation of morphologic characteristics in the latent space; 2) the ability to classify tissue type with logistic regression using latent representations, with an AUC of 0.97 and 85% accuracy, comparable to supervised deep models; 3) the ability to predict the presence of tumor in Whole Slide Images (WSIs) using multiple instance learning (MIL), achieving an AUC of 0.98 and 94% accuracy.*

*Our results show that our model captures distinct phenotypic characteristics of real tissue samples, paving the way for further understanding of tumor progression and tumor micro-environment, and ultimately refining histopathological classification for diagnosis and treatment.*

## Datasets:

### H&E Color Cancer


### H&E Breast Cancer
H&E breast cancer databases from the Netherlands Cancer Institute (NKI) cohort and the Vancouver General Hospital (VGH) cohort with 248 and 328 patients respectevely. Each of them include tissue micro-array (TMA) images, along with clinical patient data such as survival time, and estrogen-receptor (ER) status. The original TMA images all have a resolution of 1128x720 pixels, and we split each of the images into smaller patches of 224x224, and allow them to overlap by 50%. We also perform data augmentation on these images, a rotation of 90 degrees, and 180 degrees, and vertical and horizontal inversion. We filter out images in which the tissue covers less than 70% of the area. In total this yields a training set of 249K images, and a test set of 62K.

We use these Netherlands Cancer Institute (NKI) cohort and the Vancouver General Hospital (VGH) previously used in Beck et al. \[1]. These TMA images are from the [Stanford Tissue Microarray Database](https://tma.im/cgi-bin/home.pl)[2]

\[1] Beck, A.H. and Sangoi, A.R. and Leung, S. Systematic analysis of breast cancer morphology uncovers stromal features associated with survival. Science translational medicine (2018).

\[2] Robert J. Marinelli, Kelli Montgomery, Chih Long Liu, Nigam H. Shah, Wijan Prapong, Michael Nitzberg, Zachariah K. Zachariah, Gavin J. Sherlock, Yasodha Natkunam, Robert B. West, Matt van de Rijn, Patrick O. Brown, and Catherine A. Ball. The Stanford Tissue Microarray Database. Nucleic Acids Res 2008 36(Database issue): D871-7. Epub 2007 Nov 7 doi:10.1093/nar/gkm861.

You can find a pre-processed HDF5 file with patches of 224x224x3 resolution [here](https://drive.google.com/open?id=1LpgW85CVA48C8LnpmsDMdHqeCGHKsAxw), each of the patches also contains labeling information of the estrogen receptor status and survival time.

## Python Enviroment:
```
h5py                    2.9.0
numpy                   1.16.1
pandas                  0.24.1
scikit-image            0.14.2
scikit-learn            0.20.2
scipy                   1.2.0
seaborn                 0.9.0
sklearn                 0.0
tensorboard             1.12.2
tensorflow              1.12.0
tensorflow-probability  0.5.0
python                  3.6.7
```
