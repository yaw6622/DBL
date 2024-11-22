# DBL  
This is the official code repository for "A dual-branch network for crop-type mapping of scattered small agricultural fields in time series remote sensing images". {([https://doi.org/10.1016/j.rse.2024.114497](https://doi.org/10.1016/j.rse.2024.114497 "Persistent link using digital object identifier"))}  
  
## Abstract  
With the rapid advancement of remote sensing technology, the recognition of agricultural field parcels using  time-series remote sensing images has become an increasingly emphasized task. In this paper, we focus on  identifying crops within scattered, irregular, and poorly defined agricultural fields in many Asian regions. We select two representative locations with small and scattered parcels and construct two new time-series remote sensing datasets (JM dataset and CF dataset). We propose a novel deep learning model DBL, the Dual-Branch Model with Long Short-Term Memory (LSTM), which utilizes main branch and supplementary branch to accomplish accurate crop-type mapping. The main branch is designed for capturing global receptive field and the supplementary is designed for temporal and spatial feature refinement. The experiments are conducted to evaluate the performance of the DBL compared with the state-of-the-art (SOTA) models. The results indicate that the DBL model performs exceptionally well on both datasets. Especially on the CF dataset characterized by scattered and irregular plots, the DBL model achieves an overall accuracy (OA) of 97.70% and a mean intersection over union (mIoU) of 90.70%. It outperforms all the SOTA models and becomes the only model to exceed 90% mark on the mIoU score. We also demonstrate the stability and robustness of the DBL across different agricultural regions.

## Environment
Create an new conda virtual environment
```bash
conda create -n DBL python=3.9.12 -y
conda activate DBL
```
Clone this repo and install required packages:
```bash
git clone https://github.com/yaw6622/DBL
cd DBL/
pip install -r requirements.txt
```

## Train the model
```bash
cd DBL
python train.py  # Train and test DBL on the JM or CF dataset.
```

## Dataset availability
### JM & CF
The primary experiments in this project are conducted on our newly proposed datasets, **JM** and **CF**.
You can download the datasets from the following link: [JM](https://drive.google.com/file/d/1HEKonoFzjdrNUOG8AQbDiLpyTBEh1N_D/view?usp=drive_link), [CF](https://drive.google.com/file/d/1Tq6ZXakemSCTkaY0nKyRn8k9114Fpj_H/view?usp=drive_link).

### PASTIS & PASTIS-SS
In order to validate the generalization capabilities of the proposed
DBL, we conduct additional experiments on a public available dataset
PASTIS and a sub-dataset PASTIS-SS.
The complete PASTIS dataset can be downloaded form here: https://github.com/VSainteuf/pastis-benchmark
For the filtered sub-dataset PASTIS-SS, we provide the numbers of filtered time-series data. It can be found in the file PASTIS-SS.xlsx.

## Obtain the outputs  
- After trianing, you could obtain the results in './results/'

## Pretrained model
To facilitate quick usage, we provide a lightweight pretrained model for the **CF dataset** in the `pretrained_model` folder.

When using the lightweight pretrained model, please make the following adjustments to the configuration:

-   Set `self.width` to 64.
-   Change `Out_dim` from `[64, 128]` to `[[64, 64]]`.
-   Set the **LSTM hidden size** to 32.

## Acknowledgments  
  
- We thank the authors of [Utae](https://github.com/VSainteuf/utae-paps),  [CBAM](https://github.com/elbuco1/CBAM) and [Fasternet](https://github.com/JierunChen/FasterNet) for their open-source codes.
