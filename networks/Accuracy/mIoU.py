import numpy as np
import argparse
import json
from PIL import Image
from os.path import join
import rasterio

def fast_hist(a, b, n): 
   
    k = (a >= 0) & (a < n)  
    matrx=np.bincount(n * a[k].astype(int) + b[k], minlength=n ** 2)

    if len(matrx)>81:
        return matrx[len(matrx)-82:-1].reshape(n,n)
    else :
        return matrx.reshape(n,n)
 


def per_class_iu(hist):  
    return np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist)) 

def per_class_Freq(hist): 
   
    return np.sum(hist, axis=1) / np.sum(hist)

def per_class_OA(hist):
    return np.diag(hist) / np.sum(hist,axis=1)

def per_class_kappa(hist):
    hist_sum=np.sum(hist)
    p0=np.sum(np.diag(hist))/np.sum(hist)
    pe=np.sum(np.sum(hist,axis=0)*np.sum(hist,axis=1))/(hist_sum*hist_sum)
    return (p0-pe)/(1-pe)

def label_mapping(input, mapping):  
    output = np.copy(input)  
    for ind in range(len(mapping)):
        output[input == mapping[ind][0]] = mapping[ind][1]  
    return np.array(output, dtype=np.int64) 



def compute_mIoU(gt_dir, pred_dir, devkit_dir=''):  
    """
    Compute IoU given the predicted colorized images and
    """
    with open(join(devkit_dir, 'NLCD.json'), 'r') as fp: 
        info = json.load(fp)
    num_classes = int(info['classes'])  
    print('Num classes', num_classes)  
    name_classes = np.array(info['label'], dtype=str) 
    hist = np.zeros((num_classes, num_classes))

    image_path_list = join(devkit_dir, 'val.txt') 
    label_path_list = join(devkit_dir, 'label.txt')
    gt_imgs = open(label_path_list, 'r').read().splitlines()  
    gt_imgs = [join(gt_dir, x) for x in gt_imgs]  
    pred_imgs = open(image_path_list, 'r').read().splitlines()  
    # print(pred_imgs)
    pred_imgs = [join(pred_dir, x.split('/')[-1]) for x in pred_imgs]  
    pred_imgs=pred_imgs[0:-2]
    for ind in range(len(gt_imgs)-2):  
        with rasterio.open(pred_imgs[ind]) as f:
            pred = f.read(1)
            pred[np.where(pred == 5)] = 4

            print(pred_imgs[ind])
        pred = np.array(pred)
        with rasterio.open(gt_imgs[ind]) as f1:
            label = f1.read(1)
       
        label = np.array(label)
    
        if len(label.flatten()) != len(pred.flatten()): 
            print('Skipping: len(gt) = {:d}, len(pred) = {:d}, {:s}, {:s}'.format(len(label.flatten()),
                                                                                  len(pred.flatten()), gt_imgs[ind],
                                                                                  pred_imgs[ind]))
            continue
        
        hist += fast_hist(label.flatten(), pred.flatten(), num_classes) 
        if ind > 0 and ind % 10 == 0: 
            print('{:d} / {:d}: {:0.2f}'.format(ind, len(gt_imgs), 100 * np.nanmean(per_class_iu(hist))))

    mIoUs = per_class_iu(hist)  
    Freq = per_class_Freq(hist)
    Kappa = per_class_kappa(hist)
    Oa = per_class_OA(hist)
    print(Freq)
    for ind_class in range(num_classes):
        print('>' + name_classes[ind_class] + ':\t' + str(round(mIoUs[ind_class] * 100, 2))+':\t'+str(round(Freq[ind_class]* 100, 2)))
    print('> mIoU: '+str((1/4 * mIoUs[Freq > 0]).sum()))
    print('> FWIoU: '+str((Freq[Freq > 0] * mIoUs[Freq > 0]).sum()))
    print('===> OA: '+str((1/4 * Oa[Freq > 0]).sum()))
    print('===> Kappa: '+str(Kappa))# 
    return mIoUs
NY_CCLC='/home/tsf/VCNet/dataset/'
NY_Conf='/home/tsf/vitcomer/networks/Accuracy/NYconfig/'

def main():
    compute_mIoU(NY_CCLC, 
    '/path/to/the inferred results.tiff',
     NY_Conf) 
main()