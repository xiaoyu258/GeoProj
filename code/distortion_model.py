import math
import numpy as np

def distortionParameter(types):
    parameters = []
    
    if (types == 'barrel'):
        Lambda = np.random.random_sample( )* -5e-5/4
        x0 = 256
        y0 = 256
        parameters.append(Lambda)
        parameters.append(x0)
        parameters.append(y0)
        return parameters
    
    elif (types == 'pincushion'):
        Lambda = np.random.random_sample() * 8.6e-5/4
        x0 = 128
        y0 = 128
        parameters.append(Lambda)
        parameters.append(x0)
        parameters.append(y0)
        return parameters
    
    elif (types == 'rotation'):
        theta = np.random.random_sample() * 30 - 15   
        radian = math.pi*theta/180
        sina = math.sin(radian)
        cosa = math.cos(radian)
        parameters.append(sina)
        parameters.append(cosa)
        return parameters
    
    elif (types == 'shear'):
        shear = np.random.random_sample() * 0.8 - 0.4
        parameters.append(shear)
        return parameters

    elif (types == 'projective'):
    
        x1 = 0
        x4 = np.random.random_sample()* 0.1 + 0.1

        x2 = 1 - x1
        x3 = 1 - x4

        y1 = 0.005
        y4 = 1 - y1
        y2 = y1
        y3 = y4

        a31 = ((x1-x2+x3-x4)*(y4-y3) - (y1-y2+y3-y4)*(x4-x3))/((x2-x3)*(y4-y3)-(x4-x3)*(y2-y3))
        a32 = ((y1-y2+y3-y4)*(x2-x3) - (x1-x2+x3-x4)*(y2-y3))/((x2-x3)*(y4-y3)-(x4-x3)*(y2-y3))

        a11 = x2 - x1 + a31*x2
        a12 = x4 - x1 + a32*x4
        a13 = x1

        a21 = y2 - y1 + a31*y2
        a22 = y4 - y1 + a32*y4
        a23 = y1
       
        parameters.append(a11)
        parameters.append(a12)
        parameters.append(a13)
        parameters.append(a21)
        parameters.append(a22)
        parameters.append(a23)
        parameters.append(a31)
        parameters.append(a32)
        return parameters
    
    elif (types == 'wave'):
        mag = np.random.random_sample() * 32
        parameters.append(mag)
        return parameters


def distortionModel(types, xd, yd, W, H, parameter):
    
    if (types == 'barrel' or types == 'pincushion'):
        Lambda = parameter[0]
        x0    = parameter[1]
        y0    = parameter[2]
        coeff = 1 + Lambda * ((xd - x0)**2 + (yd - y0)**2)
        if (coeff == 0):
            xu = W
            yu = H
        else:
            xu = (xd - x0)/coeff + x0
            yu = (yd - y0)/coeff + y0
        return xu, yu
    
    elif (types == 'rotation'):
        sina  = parameter[0]
        cosa  = parameter[1]
        xu =  cosa*xd + sina*yd + (1 - sina - cosa)*W/2
        yu = -sina*xd + cosa*yd + (1 + sina - cosa)*H/2
        return xu, yu
    
    elif (types == 'shear'):
        shear = parameter[0]
        xu =  xd + shear*yd - shear*W/2
        yu =  yd 
        return xu, yu
    
    elif (types == 'projective'):
        a11 = parameter[0]
        a12 = parameter[1]
        a13 = parameter[2]
        a21 = parameter[3]
        a22 = parameter[4]
        a23 = parameter[5]
        a31 = parameter[6]
        a32 = parameter[7]
        im = xd/(W - 1.0)
        jm = yd/(H - 1.0)
        xu = (W - 1.0) *(a11*im + a12*jm +a13)/(a31*im + a32*jm + 1)
        yu = (H - 1.0)*(a21*im + a22*jm +a23)/(a31*im + a32*jm + 1)
        return xu, yu
    
    elif (types == 'wave'):
        mag = parameter[0]
        yu = yd
        xu = xd + mag*math.sin(math.pi*4*yd/W)
        return xu, yu
        
        