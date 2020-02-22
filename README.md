# An extended-2D CNN approach for diagnosis of Alzheimer's disease through structural MRI

## Description:
This repository is a combination of scripts generated throughout my Master's thesis, entitled "An extended-2D CNN approach for diagnosis of Alzheimer's disease through structural MRI". This work yielded the following two distinct publications.


In order to extract relevant volumetric information, we proposed the extended-2D data, which is performed in three main steps:
![image](https://github.com/mariecpereira/extension-2D-CNN/blob/master/images/25D_modf1.png =100)
First, we selected N center slices in the coronal plane, then, every second slice is selected. Secondly, to compose each 3-channel image, we select two more slices spaced by n millimeters. These slices, spaced by n are then the new extended-2D image. For example, if the first selection is the slice 102, then the slice 98 and the slice 106 will compose the new image. By doing this, we take volumetric changes within slices.


The first publication summarizes an ensemble of different architectures taking into account the coordinates of each patch as additional information for the network:

- Pereira, M. et al., An extended-2D CNN approach for diagnosis of Alzheimer's disease through structural MRI. In International Society for Magnetic Resonance in Medicine (ISMRM) Annual Meeting and Exhibition 2019

This work summarizes an end-to-end solution for classification, using only extended-2D data:

- Pereira, M. et al., An extended-2D CNN for multiclass Alzheimer's Disease diagnosis through Structural MRI. In SPIE Medical Imaging 2020: Computer-Aided Diagnosis

For a full description of the methods, you may also check the full thesis [here](http://repositorio.unicamp.br/bitstream/REPOSIP/334116/1/Pereira_MarianaEugeniaDeCarvalho_M.pdf).

## Requirements:
- Python 3.5.4
- Numpy 1.12.1
- Scipy 1.0.1
- Matplotlib 2.0.2
- Scikit-learn 0.19.0
- Nibabel 2.2.1
- Pytorch 1.4

## Dataset:
Data used in preparation of this article were obtained from the Alzheimer's Disease Neuroimaging Initiative (ADNI) database (adni.loni.usc.edu). As such, the investigators within the ADNI contributed to the design and implementation of ADNI and/or provided data but did not participate in analysis or writing of this report. A complete listing of ADNI investigators can be found at: http://adni.loni.usc.edu/wp-content/uploads/how_to_apply/ADNI_Acknowledgement_List.pdf
