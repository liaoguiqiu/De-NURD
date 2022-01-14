# De-NURD
This package provides an implementation of the inference and training data generation pipeline of A two-branch Nonuiform rotational distortion (NURD) compensation networks for rotational scanning catheter,
This is the more acurate version that relys on Correlation Matrix, in comparison to end-to-end version
. For simplicity the model in referred as De-NURD nets.
## First time setup

The following steps are required in order to run CEnet:

1.  Install deep learning python packages(either linux or windows) .
    *   Recomended: Install anaconda
        [anaconda](https://www.anaconda.com/).
    *   install pytorch
        [pytorch](https://pytorch.org/).
    *   install mission packages (i.e. cv2, seaborn..)
        [pip install packages](https://packaging.python.org/en/latest/tutorials/installing-packages/)
        or
        [conda install packages](https://docs.anaconda.com/anaconda/user-guide/tasks/install-packages/)
1.  Organize the raw  dataset 
	
	folder structure should be like this:

	-Root 

		--Root/dataset/

			---Root/dataset/for IVUS

				---Root/dataset/for IVUS/train

					---Root/dataset/for IVUS/train/img

					---Root/dataset/for IVUS/train/label



				---Root/dataset/for IVUS/test

					---Root/dataset/for IVUS/test/label

					---Root/dataset/for IVUS/test/label


		--Root/out

1.  Download [pre-trained model parameters] (https://seafile.unistra.fr/d/54c9103c11f142ae9dc7/)

  
 
 
## Key useful tools/python scripts
To run any runable scripts in this project,
if using the visual studio (not vs code) or pycharm, it can be run directly. 
If using vs code, please run scripts as module.
If run in terminal of linux or anaconda prompt, run as module, using following command: python3 -m subfoler.scripts(i.e. python3 -m tool.convert)

1.    toolfolder:


	*   cover_raw_data.py: convert raw data to fixed format for algorithm

 


1.  In main folder/project folder  : 

    most of these scripts are used to predict contour, specifically:
 
	*   Dataset Generator.py is used to generate separated image pairs and image arrays for training two branches

	*   Dataset Generator_OLG.py is an exension of the Dataset Generator.py that deploy to generate the data on the fly

	*   Crrect_sequence_integral.py is used to correct the NURD by a integral estimation scheme

	*   Crrect_sequence_iteration.py is used to correct the NURD with a iterative estimation scheme

