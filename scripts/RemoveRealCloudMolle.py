import cv2
import torch
import numpy as np
from pathlib import Path
from src.config_default import TrainingConfig
from src.models.UnetPartialConvModel import generator, Wgangenerator
from src.models.UnetPartialConvModelNIR import generatorNIR,Wgangenerator
from src.shared.visualization import normalize_batch_tensor
from src.shared.convert import convertToFloat32
from src.shared.convert import _normalize
from src.shared.modelUtility import modelHelper
from src.evalMetrics.eval_helper import remove_outliers

import joblib

if __name__ == '__main__':


    # Read image

    pts = []


    ## From https://www.programmersought.com/article/3449903953/
    def draw_roi(event, x, y, flags, param):
        img2 = img.copy()

        if event == cv2.EVENT_LBUTTONDOWN:
            pts.append((x, y))

        if event == cv2.EVENT_RBUTTONDOWN:
            pts.pop()

        if event == cv2.EVENT_MBUTTONDOWN:
            mask = np.zeros(img.shape, np.uint8)
            points = np.array(pts, np.int32)
            points = points.reshape((-1, 1, 2))
            mask = cv2.polylines(mask, [points], True, (255, 255, 255), 2)
            mask2 = cv2.fillPoly(mask.copy(), [points], (255, 255, 255))
            mask3 = cv2.fillPoly(mask.copy(), [points], (0, 255, 0))

            show_image = cv2.addWeighted(src1=img, alpha=0.8, src2=mask3, beta=0.2, gamma=0)

            cv2.imshow("mask", mask2)
            cv2.imshow("show_img", show_image)

            ROI = cv2.bitwise_and(mask2, img)
            cv2.imshow("ROI", ROI)
            cv2.waitKey(0)

        if len(pts) > 0:
            cv2.circle(img2, pts[-1], 3, (0, 0, 255), -1)

        if len(pts) > 1:
            # 画线
            for i in range(len(pts) - 1):
                cv2.circle(img2, pts[i], 5, (0, 0, 255), -1)
                cv2.line(img=img2, pt1=pts[i], pt2=pts[i + 1], color=(255, 0, 0), thickness=2)

        cv2.imshow('image', img2)

    #C:\Users\Morten From\Downloads\raw_cloud\raw_cloud
    #T35UNB_20200617T092029_train_1450
    img = cv2.imread(
        r"C:\Users\Morten From\Downloads\raw_cloud\raw_cloud\T35UNB_20200617T092029_train_1450")
    model_path = (
        r"C:\Users\panda\PycharmProjects\Image_Inpainting_Sat\Master_Satelite_Image_Inpainting\OutputModels\PartialConvolutionsWgan_901.pt")
    local_test_path = Path(r"C:\Users\Morten From\Downloads\raw_cloud\raw_cloud")
    cv2.namedWindow('image')
    cv2.setMouseCallback('image', draw_roi)
    while True:
        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            break
        if key == ord("s"):
            saved_data = {
                "ROI": pts
            }
            joblib.dump(value=saved_data, filename="config.pkl")
            break
    cv2.destroyAllWindows()
    mask = np.zeros(img.shape, np.uint8)
    points = np.array(pts, np.int32)
    points = points.reshape((-1, 1, 2))
    mask = cv2.polylines(mask, [points], True, (255, 255, 255), 2)
    mask2 = cv2.fillPoly(mask.copy(), [points], (255, 255, 255))
    mask2[mask2==0]=1
    mask2[mask2==255]=0

    img = cv2.imread(
        r"C:\Users\panda\PycharmProjects\Image_Inpainting_Sat\Master_Satelite_Image_Inpainting\data\processed\Remove_cloud\T35UNB_20200617T092029\bandTCIRGB\Test\RGBImages\original_0RGB\T35UNB_20200617T092029_test_1450.tiff",-1)
    config = TrainingConfig()
    image = convertToFloat32(img)
    image = remove_outliers(image)
    image = torch.from_numpy(np.array(image).astype(np.float32)).transpose(0, 1).transpose(0, 2).contiguous()
    image = image.to(config.device)
    masks = torch.from_numpy(mask2).transpose(0, 1).transpose(0, 2).contiguous()
    masks = masks.type(torch.cuda.FloatTensor)
    #masks = 1 - masks
    masks.to(config.device)

    # Load the model
    gen = generator().to(config.device)
    gen.load_state_dict(torch.load(model_path))  ## Use epochs to identify model number
    gen.eval()
    image = image.unsqueeze(0)
    masks= masks.unsqueeze(0)
    fake_masked_images = torch.mul(image, masks)
    generated_images = gen(image, masks)
    modelHelper.save_tensor_batch(image, fake_masked_images, generated_images, config.batch_size,
                                  Path.joinpath(local_test_path, "real_clod" ))

    #cv2.imshow("Image", imCrop)
