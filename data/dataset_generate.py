import numpy as np
import skimage
import skimage.io as io
from skimage.transform import rescale
import scipy.io as scio
import distortion_model

sourcePath = '/home/xliea/GenerateData256/dataset512'


trainDisPath = '/home/xliea/GenerateData256/GeneralDataset256/train/distorted'
trainUvPath  = '/home/xliea/GenerateData256/GeneralDataset256/train/uv'

testDisPath = '/home/xliea/GenerateData256/GeneralDataset256/test/distorted'
testUvPath  = '/home/xliea/GenerateData256/GeneralDataset256/test/uv'

trainNum = 50000
testNum  = 5000

def generatedata(types, k, trainFlag):
    
    print(types,trainFlag,k)
    
    width  = 512
    height = 512

    parameters = distortion_model.distortionParameter(types)
    
    OriImg = io.imread('%s%s%s%s' % (sourcePath, '/', str(k).zfill(6), '.jpg'))

    disImg = np.array(np.zeros(OriImg.shape), dtype = np.uint8)
    u = np.array(np.zeros((OriImg.shape[0],OriImg.shape[1])), dtype = np.float32)
    v = np.array(np.zeros((OriImg.shape[0],OriImg.shape[1])), dtype = np.float32)
    
    cropImg = np.array(np.zeros((height/2,width/2,3)), dtype = np.uint8)
    crop_u  = np.array(np.zeros((height/2,width/2)), dtype = np.float32)
    crop_v  = np.array(np.zeros((height/2,width/2)), dtype = np.float32)
    
    # crop range
    xmin = width*1/4
    xmax = width*3/4 - 1
    ymin = height*1/4
    ymax = height*3/4 - 1

    for i in range(width):
        for j in range(height):
            
            xu, yu = distortion_model.distortionModel(types, i, j, width, height, parameters)
            
            if (0 <= xu < width - 1) and (0 <= yu < height - 1):

                u[j][i] = xu - i
                v[j][i] = yu - j
                
                # Bilinear interpolation
                Q11 = OriImg[int(yu), int(xu), :]
                Q12 = OriImg[int(yu), int(xu) + 1, :]
                Q21 = OriImg[int(yu) + 1, int(xu), :]
                Q22 = OriImg[int(yu) + 1, int(xu) + 1, :]
                
                disImg[j,i,:] = Q11*(int(xu) + 1 - xu)*(int(yu) + 1 - yu) + \
                                 Q12*(xu - int(xu))*(int(yu) + 1 - yu) + \
                                 Q21*(int(xu) + 1 - xu)*(yu - int(yu)) + \
                                 Q22*(xu - int(xu))*(yu - int(yu))

                            
                if(xmin <= i <= xmax) and (ymin <= j <= ymax):
                    cropImg[j - ymin, i - xmin, :] = disImg[j,i,:]
                    crop_u[j - ymin, i - xmin] = u[j,i]
                    crop_v[j - ymin, i - xmin] = v[j,i]
                    
    if trainFlag == True:
        saveImgPath =  '%s%s%s%s%s%s' % (trainDisPath, '/',types,'_', str(k).zfill(6), '.jpg')
        saveMatPath =  '%s%s%s%s%s%s' % (trainUvPath, '/',types,'_', str(k).zfill(6), '.mat')
        io.imsave(saveImgPath, cropImg)
        scio.savemat(saveMatPath, {'u': crop_u,'v': crop_v})  
    else:
        saveImgPath =  '%s%s%s%s%s%s' % (testDisPath, '/',types,'_', str(k).zfill(6), '.jpg')
        saveMatPath =  '%s%s%s%s%s%s' % (testUvPath, '/',types,'_', str(k).zfill(6), '.mat')
        io.imsave(saveImgPath, cropImg)
        scio.savemat(saveMatPath, {'u': crop_u,'v': crop_v})   
        
def generatepindata(types, k, trainFlag):
    
    print(types,trainFlag,k)
    
    width  = 256
    height = 256

    parameters = distortion_model.distortionParameter(types)
    
    OriImg = io.imread('%s%s%s%s' % (sourcePath, '/', str(k).zfill(6), '.jpg'))
    temImg = rescale(OriImg, 0.5, mode='reflect')
    ScaImg = skimage.img_as_ubyte(temImg)
    
    padImg = np.array(np.zeros((ScaImg.shape[0] + 1,ScaImg.shape[1] + 1, 3)), dtype = np.uint8)
    padImg[0:height, 0:width, :] = ScaImg[0:height, 0:width, :]
    padImg[height, 0:width, :] = ScaImg[height - 1, 0:width, :]
    padImg[0:height, width, :] = ScaImg[0:height, width - 1, :]
    padImg[height, width, :] = ScaImg[height - 1, width - 1, :]

    disImg = np.array(np.zeros(ScaImg.shape), dtype = np.uint8)
    u = np.array(np.zeros((ScaImg.shape[0],ScaImg.shape[1])), dtype = np.float32)
    v = np.array(np.zeros((ScaImg.shape[0],ScaImg.shape[1])), dtype = np.float32)

    for i in range(width):
        for j in range(height):
            
            xu, yu = distortion_model.distortionModel(types, i, j, width, height, parameters)
            
            if (0 <= xu <= width - 1) and (0 <= yu <= height - 1):

                u[j][i] = xu - i
                v[j][i] = yu - j
                
                # Bilinear interpolation
                Q11 = padImg[int(yu), int(xu), :]
                Q12 = padImg[int(yu), int(xu) + 1, :]
                Q21 = padImg[int(yu) + 1, int(xu), :]
                Q22 = padImg[int(yu) + 1, int(xu) + 1, :]
                
                disImg[j,i,:] = Q11*(int(xu) + 1 - xu)*(int(yu) + 1 - yu) + \
                                 Q12*(xu - int(xu))*(int(yu) + 1 - yu) + \
                                 Q21*(int(xu) + 1 - xu)*(yu - int(yu)) + \
                                 Q22*(xu - int(xu))*(yu - int(yu))
    
    if trainFlag == True:
        saveImgPath =  '%s%s%s%s%s%s' % (trainDisPath, '/',types,'_', str(k).zfill(6), '.jpg')
        saveMatPath =  '%s%s%s%s%s%s' % (trainUvPath, '/',types,'_', str(k).zfill(6), '.mat')
        io.imsave(saveImgPath, disImg)
        scio.savemat(saveMatPath, {'u': u,'v': v})  
    else:
        saveImgPath =  '%s%s%s%s%s%s' % (testDisPath, '/',types,'_', str(k).zfill(6), '.jpg')
        saveMatPath =  '%s%s%s%s%s%s' % (testUvPath, '/',types,'_', str(k).zfill(6), '.mat')
        io.imsave(saveImgPath, disImg)
        scio.savemat(saveMatPath, {'u': u,'v': v}) 
        
        
for types in ['barrel','rotation','shear','wave']: 
    for k in range(trainNum):
        generatedata(types, k, trainFlag = True)

    for k in range(trainNum, trainNum + testNum):
        generatedata(types, k, trainFlag = False)
        
for types in ['pincushion','projective']: 
    for k in range(trainNum):
        generatepindata(types, k, trainFlag = True)

    for k in range(trainNum, trainNum + testNum):
        generatepindata(types, k, trainFlag = False)
