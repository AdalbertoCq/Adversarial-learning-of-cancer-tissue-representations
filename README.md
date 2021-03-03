# Adversarial learning of cancer tissue representations
 
**Abstract:**

*Deep learning based analysis of histopathology images shows promise in advancing understanding of tumor progression, tumor micro-environment, and their underpinning biological processes. So far, these approaches have focused on extracting information associated with annotations. In this work, we ask how much information can be learned from the tissue architecture itself. 

We present an Adversarial learning model to extract feature representations of cancer tissue, without the need for manual annotations. We show that these representations are able to identify a variety of morphological characteristics across three cancer types: Breast, colon, and lung. This is supported by 1) the separation of morphologic characteristics in the latent space; 2) the ability to classify tissue type with logistic regression using latent representations, with an AUC of 0.97 and 85% accuracy, comparable to supervised deep models; 3) the ability to predict the presence of tumor in Whole Slide Images (WSIs) using Multiple Instance Learning (MIL), achieving an AUC of 0.98 and 94% accuracy.

Our results show that our model captures distinct phenotypic characteristics of real tissue samples, paving the way for further understanding of tumor progression and tumor micro-environment, and ultimately refining histopathological classification for diagnosis and treatment.*
