# Adversarial learning of cancer tissue representations
 
**Abstract:**

*Deep learning based analysis of histopathology images shows promise in advancing understanding of tumor progression, tumor micro-environment, and their underpinning biological processes. So far, these approaches have focused on extracting information associated with annotations. In this work, we ask how much information can be learned from the tissue architecture itself.*

*We present an adversarial learning model to extract feature representations of cancer tissue, without the need for manual annotations. We show that these representations are able to identify a variety of morphological characteristics across three cancer types: Breast, colon, and lung. This is supported by 1) the separation of morphologic characteristics in the latent space; 2) the ability to classify tissue type with logistic regression using latent representations, with an AUC of 0.97 and 85% accuracy, comparable to supervised deep models; 3) the ability to predict the presence of tumor in Whole Slide Images (WSIs) using multiple instance learning (MIL), achieving an AUC of 0.98 and 94% accuracy.*

*Our results show that our model captures distinct phenotypic characteristics of real tissue samples, paving the way for further understanding of tumor progression and tumor micro-environment, and ultimately refining histopathological classification for diagnosis and treatment.*

## Citation
```
@InProceedings{quiros2021adversarial,
  title={Adversarial learning of cancer tissue representations},
  author={Quiros, Adalberto Claudio and Coudray, Nicolas and Yeaton, Anna and Suhnhem, Wisuwat and Murray-Smith, Roderick and Tsirigos, Aristotelis and Yuan, Ke},
  booktitle={International Conference on Medical Image Computing and Computer-Assisted Intervention},
  year={2021},
  organization={Springer}
}
```

## Training the model:
You can find a pre-processed HDF5 file with patches of 224x224x3 resolution of the H&E breast cancer dataset [here](https://drive.google.com/open?id=1LpgW85CVA48C8LnpmsDMdHqeCGHKsAxw). Place the 'vgh_nki' under the 'datasets' folder in the main 'Adversarial-learning-of-cancer-tissue-representations' path.

Each model was trained on an NVIDIA Titan 24 GB for 45 epochs, approximately 72 hours.

```
usage: run_pathgan_encoder.py [-h] [--model MODEL] [--img_size IMG_SIZE]
                              [--img_ch IMG_CH] [--dataset DATASET]
                              [--marker MARKER] [--z_dim Z_DIM]
                              [--epochs EPOCHS] [--batch_size BATCH_SIZE]
                              [--check_every CHECK_EVERY] [--restore]
                              [--report] [--main_path MAIN_PATH]
                              [--dbs_path DBS_PATH]

PathologyGAN Encoder trainer.

optional arguments:
  -h, --help                    show this help message and exit
  --model MODEL                 Model name.
  --img_size IMG_SIZE           Image size for the model.
  --img_ch IMG_CH               Number of channels for the model.
  --dataset DATASET             Dataset to use.
  --marker MARKER               Marker of dataset to use.
  --z_dim Z_DIM                 Latent space size.
  --epochs EPOCHS               Number epochs to run: default is 45 epochs.
  --batch_size BATCH_SIZE       Batch size, default size is 64.
  --check_every CHECK_EVERY     Save checkpoint and generate samples every X epcohs.
  --restore                     Restore previous run and continue.
  --report                      Report latent space figures.
  --main_path MAIN_PATH         Path for the output run.
  --dbs_path DBS_PATH           Directory with DBs to use.
```

* Command example:
```
python3 run_pathgan_encoder.py 
```

## Projecting images onto the latent space:
Once you have a trained model you can project images into the latent space, the vector represenetation will be placed in 'results' folder on an H5 file.
```
usage: project_real_tissue_latent_space.py [-h] --checkpoint CHECKPOINT
                                           --real_hdf5 REAL_HDF5
                                           [--batch_size BATCH_SIZE]
                                           [--z_dim Z_DIM] [--model MODEL]
                                           [--img_size IMG_SIZE]
                                           [--img_ch IMG_CH]
                                           [--dataset DATASET]
                                           [--marker MARKER]
                                           [--dbs_path DBS_PATH]
                                           [--main_path MAIN_PATH]
                                           [--num_clusters NUM_CLUSTERS]
                                           [--clust_percent CLUST_PERCENT]
                                           [--features] [--save_img]

Projection of tissue images onto the GAN's latent space.

optional arguments:
  -h, --help                      show this help message and exit
  --checkpoint CHECKPOINT         Path to pre-trained weights (.ckt) of PathologyGAN.
  --real_hdf5 REAL_HDF5           Path for real image to encode.
  --batch_size BATCH_SIZE         Batch size.
  --z_dim Z_DIM                   Latent space size.
  --model MODEL                   Model name.
  --img_size IMG_SIZE             Image size for the model.
  --img_ch IMG_CH                 Image channels for the model.
  --dataset DATASET               Dataset to use.
  --marker MARKER                 Marker of dataset to use.
  --dbs_path DBS_PATH             Directory with DBs to use.
  --main_path MAIN_PATH           Path for the output run.
  --features                      Flag to run features over the images.
  --save_img                      Save reconstructed images in the H5 file.
```

## Datasets:
### H&E Breast Cancer
H&E breast cancer databases from the Netherlands Cancer Institute (NKI) cohort and the Vancouver General Hospital (VGH) cohort with 248 and 328 patients respectevely. Each of them include tissue micro-array (TMA) images, along with clinical patient data such as survival time, and estrogen-receptor (ER) status. The original TMA images all have a resolution of 1128x720 pixels, and we split each of the images into smaller patches of 224x224, and allow them to overlap by 50%. We also perform data augmentation on these images, a rotation of 90 degrees, and 180 degrees, and vertical and horizontal inversion. We filter out images in which the tissue covers less than 70% of the area. In total this yields a training set of 249K images, and a test set of 62K.

We use these Netherlands Cancer Institute (NKI) cohort and the Vancouver General Hospital (VGH) previously used in Beck et al. \[1]. These TMA images are from the [Stanford Tissue Microarray Database](https://tma.im/cgi-bin/home.pl)[2]

You can find a pre-processed HDF5 file with patches of 224x224x3 resolution [here](https://drive.google.com/open?id=1LpgW85CVA48C8LnpmsDMdHqeCGHKsAxw), each of the patches also contains labeling information of the estrogen receptor status and survival time.

### H&E Colorectal Cancer
The H&E colorectal cancer dataset can be found [here](https://zenodo.org/record/1214456#.XyAAxPhKgkg). The dataset from National Center for Tumor diseases (NCT, Germany) [3] provides tissue images of 224×224 resolution with an as- sociated type of tissue label: Adipose, background, debris, lymphocytes, mucus, smooth muscle, normal colon mucosa, cancer-associated stroma, and colorectal adenocarcinoma epithelium (tumor). The dataset is divided into a training set of 100K tissue tiles and 86 patients, and a test set of 7K tissue tiles and 50 patients, there is no overlapping patients between train and test sets. 

### H&E Lung Cancer
The H&E lung cancer dataset can be found at [The Cancer Genome Atlas (TCGA)](https://portal.gdc.cancer.gov). It contains samples with adenocarcinoma (LUAD), squamous cell carcinoma (LUSC), and normal tissue, composed by 1807 Whole Slide Images (WSIs) of 1184 patients. We make use of the pipeline provided in Coudray et al. [4],  diving each WSI into patches of 224x224 and filtering out images with less than 50% tissue in total area and apply stain normalization [5]. In addition, we label each slide as tumor and non-tumor depending on the presence of lung cancer in the tissue. Finally, we split the dataset into a training set of 916K tissue patches and 666 patients, and a test set of 569K tissue patches and 518 patients, with no overlapping patients between both sets. We use this dataset to apply multiple instance learning (MIL) over latent representations, testing the performance to predict the presence of tumor in the WSI.

\[1] Beck, A.H. and Sangoi, A.R. and Leung, S. Systematic analysis of breast cancer morphology uncovers stromal features associated with survival. Science translational medicine, 2018.

\[2] Robert J. Marinelli, Kelli Montgomery, Chih Long Liu, Nigam H. Shah, Wijan Prapong, Michael Nitzberg, Zachariah K. Zachariah, Gavin J. Sherlock, Yasodha Natkunam, Robert B. West, Matt van de Rijn, Patrick O. Brown, and Catherine A. Ball. The Stanford Tissue Microarray Database. Nucleic Acids Res 2008 36(Database issue): D871-7. Epub 2007.

\[3] Kather, J.N., Halama, N., Marx, A.: 100,000 histological images of human colorectal cancer and healthy tissue, 2018.

\[4] Coudray, N., Ocampo, P.S., Sakellaropoulos, T., Narula, N., Snuderl, M., Fenyo ̈, D., Moreira, A.L., Razavian, N., Tsirigos, A.: Classification and mutation predic- tion from non–small cell lung cancer histopathology images using deep learning. Nature Medicine, 2018.

\[5] Reinhard, E., Adhikhmin, M., Gooch, B., Shirley, P.: Color transfer be- tween images. IEEE Computer Graphics and Applications, 2001.


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
