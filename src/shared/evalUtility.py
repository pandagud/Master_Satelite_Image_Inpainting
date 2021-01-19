from pathlib import Path
from datetime import datetime
import time
def saveEvalToTxt(model_name,PSNR,minPSNR,maxPSNR, SSIM,minSSSIM,maxSSIM, CC,minCC,maxCC, FID, time, path):
    # Function to save to txt file.
    filename = Path.joinpath(path, model_name + '_EvaluationMetrics.txt')
    # Creates file if it does not exist, else does nothing

    if not filename.parent.exists():
        filename.parent.mkdir()
    filename.touch(exist_ok=True)
    # Current time
    now = datetime.now()
    dt_string = now.strftime("%d_%m_%Y_%H_%M_%S")
    # then open, write and close file again
    file = open(filename, 'a+')
    file.write('For generated images in folder: ' + str(
        path) + '\n' + ' The current time for this run is '+str(dt_string)+ '\n'+'The output evaluation metrics are' +
               '\n' + 'mean PSNR: ' + str(PSNR) +  'min PSNR: ' + str(minPSNR) + 'max PSNR: ' + str(maxPSNR) +'\n' +
               'SSIM: ' + str(SSIM) + 'min SSIM: ' + str(minSSSIM) + 'max SSIM: ' + str(maxSSIM)+'\n' +
               'CC: ' + str(CC) + 'min CC: ' + str(minCC) + 'max CC: ' + str(maxCC) + '\n' + 'FID: ' +
               str(FID) + '\n' + 'Time: ' + str(time) + '\n')
    file.close()
def saveEvalToTxt(model_name,MAE,minMAE,maxMAE,SDD,minSDD,maxSDD,SSIM,minSSIM,maxSSIM,SCISSIM,minSCISSIM,maxSCISSIM,PSNR,minPSNR,maxPSNR,CC,minCC,maxCC,RMSE,minRMSE,maxRMSE, FID, time, path):
    # Function to save to txt file.
    filename = Path.joinpath(path, model_name + '_EvaluationMetrics.txt')
    # Creates file if it does not exist, else does nothing

    if not filename.parent.exists():
        filename.parent.mkdir()
    filename.touch(exist_ok=True)
    # Current time
    now = datetime.now()
    dt_string = now.strftime("%d_%m_%Y_%H_%M_%S")
    # then open, write and close file again
    file = open(filename, 'a+')
    file.write('For generated images in folder: ' + str(
        path) + '\n' + 'The current time for this run is '+str(dt_string)+ '\n'+'The output evaluation metrics are' +
               '\n' + 'mean MAE: ' + str(MAE) + ' min MAE: ' + str(minMAE) + ' max MAE: ' + str(maxMAE) + '\n' +
               '\n' + 'mean SDD: ' + str(SDD) + ' min SDD: ' + str(minSDD) + ' max SDD: ' + str(maxSDD) + '\n' +
               '\n' + 'mean PSNR: ' + str(PSNR) + ' min PSNR: ' + str(minPSNR) + ' max PSNR: ' + str(maxPSNR) +'\n' +
               '\n' +'SSIM: ' + str(SSIM) + ' min SSIM: ' + str(minSSIM) + ' max SSIM: ' + str(maxSSIM)+'\n' +
               '\n' + 'SSIM_scikitImage: ' + str(SCISSIM) + ' min SSIM_scikitImage: ' + str(minSCISSIM) + ' max SSIM_scikitImage: ' + str(maxSCISSIM) + '\n' +
               '\n' +'CC: ' + str(CC) + ' min CC: ' + str(minCC) + ' max CC: ' + str(maxCC) +'\n' +
               '\n' +'mean RMSE: ' + str(RMSE) + ' min RMSE: ' + str(minRMSE) + ' max RMSE: ' + str(maxRMSE) +'\n' +
               '\n' + 'FID: ' +str(FID) + '\n' + 'Time: ' + str(time) + '\n')
    file.close()

