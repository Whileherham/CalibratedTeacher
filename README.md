## Calibrated Teacher for Sparsely Annotated Object Detection
This is the official implementation of our paper Calibrated Teacher for Sparsely Annotated Object Detection (AAAI 23)  
The codes are implemented based on SoftTeacher(https://github.com/microsoft/SoftTeacher) and https://github.com/EFS-OpenSource/calibration-framework#id71
Thanks for their great work!

## Requirements
- PyTorch 1.10.1
- mmcv-full 1.4.5
- mmdet 2.19.0 (https://github.com/open-mmlab/mmdetection)
- netcal 1.2.1 (https://github.com/EFS-OpenSource/calibration-framework#id71)  

## Usage
### 1 Calibrated Teacher (without fiou)
> ```bash
> bash ./tools/dist_train.sh configs/WHH_cali_retina/cali_04_07.py 8 
> --work-dir path_to_calibratedteacher 
> ```
> ### 2 Add FIoU (resume from 152000 iter)
> ```bash
> bash ./tools/dist_train.sh configs/WHH_cali_retina_focaliou/cali_04_07_focaldefault.py 8 
> --work-dir path_to_calibratedteacher 
> ```
## Dataset
Coming soon

## Others
Thanks for your interests! I feel sorry for releasing the code so late. **I have to finish the Tencent open-source review process, then work on my thesis and apply for a job**.  
Since I'm going to work soon, I don't have time to carefully organize the original code. I only delete the private information and irrelevant code, and thus you may see **some ridiculous variable names and inelegant implementation**. 
Besides I rent a house in a remote suburb, no money for broadband. Therefore I cannot upload the dataset and checkpoints right now, or I will be out of data :( **I will try to find a free and reliable wifi**.  
I have not re-run the project after deleting so much code. So if any problems left, please contact me at wang-hh20@tsinghua.org.cn

 