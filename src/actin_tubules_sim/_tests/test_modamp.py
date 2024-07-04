from actin_tubules_sim import make_matrix, create_psf, apodize, makeoverlaps, fitk0andmodamps
import pytest 
import numpy as np
from scipy.misc import face
from scipy import ndimage 
import matplotlib.pyplot as plt 
from csbdeep.utils import normalize

@pytest.mark.parametrize("Nx", [128])
@pytest.mark.parametrize("Ny", [128])
@pytest.mark.parametrize("wavelength", [0.488,0.560])
@pytest.mark.parametrize("dxy", [62.6e-3])
@pytest.mark.parametrize("ndirs", [3])
@pytest.mark.parametrize("nphases",[3])
def test_cal_modamp(
    Nx, Ny, wavelength, dxy, ndirs, nphases
):
    image = face(gray=True)
    print(image.shape)
    zoom_factors = (Nx * ndirs / image.shape[0], Ny * nphases / image.shape[1])
    image = ndimage.zoom(image, zoom_factors)
    norders = int((nphases + 1) / 2)
    excNA = 1.35
    scale_gt = 2.0
    setupNUM = 1
    space = wavelength/excNA
    k0mod = 1 / space
    napodize = 10
    nphases = 3
    ndirs = 3
    sigma_x = 1
    sigma_y = 1
    recalcarrays = 2
    ifshowmodamp = 0
    k0angle_c = [1.48, 2.5272, 3.5744]
    k0angle_g = [0.0908, -0.9564, -2.0036] 
    [Nx_hr, Ny_hr] = [Nx* scale_gt, Ny* scale_gt] 
    [dx_hr, dy_hr] = [x / scale_gt for x in [dxy, dxy]]
    xx = dx_hr * np.arange(-Nx_hr / 2, Nx_hr / 2, 1)
    yy = dy_hr * np.arange(-Ny_hr / 2, Ny_hr / 2, 1)
    [X, Y] = np.meshgrid(xx, yy)
    
    dkx = 1.0 / ( Nx *  dxy)
    dky = 1.0 / ( Ny * dxy)
    PSF, OTF = create_psf(sigma_x, 
                    sigma_x,
                    Nx_hr, 
                    Ny_hr, 
                    dkx, 
                    dky)
    parameters = {
    'Ny': Ny,
    'Nx': Nx,
    'wavelength':wavelength,
    'excNA':excNA,
    'ndirs':ndirs,
    'nphases':nphases,
    'ifshowmodamp':ifshowmodamp,
    'scale_gt': scale_gt,
    'setupNUM': setupNUM,
    'k0angle_c':k0angle_c,
    'k0angle_g':k0angle_g,
    'recalcarrays':recalcarrays,
    'dxy':dxy,
    'space':space,
    'k0mod':k0mod,
    'norders':norders,
    'napodize':napodize,
    'scale': scale_gt,
    'sigma_x': sigma_x,
    'sigma_y': sigma_y,

    
    }
    
    cur_k0, modamp = local_cal_modamp(image, OTF, parameters)
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.title("Original Image")
    plt.imshow(image, cmap='gray')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.title(f"cur_k0_{cur_k0.shape}")
    plt.imshow(cur_k0, cmap='gray')
    plt.axis('off')

    plt.show()
    print(type(cur_k0), type(modamp))
    
def local_cal_modamp(image, OTF, parameters):
    # Parameters initialization
    Nx, Ny = parameters['Nx'], parameters['Ny']
    lamda = parameters['wavelength']
    dxy = parameters['dxy']
    ndirs, nphases = parameters['ndirs'], parameters['nphases']
    norders, napodize = parameters['norders'], parameters['napodize']
    k0mod, k0angle = parameters['k0mod'], parameters['k0angle_c']
    k0 = np.transpose([k0mod * np.cos(k0angle), k0mod * np.sin(k0angle)])
    # Calculate modamp
    image = np.reshape(image, [Ny, Nx, ndirs, nphases])
    modamp = []
    cur_k0 = np.copy(k0)
    for d in range(ndirs):
        sepMatrix = make_matrix(nphases, norders)
        imagePro = np.squeeze(image[:, :, d, :])
        imagePro = np.transpose(imagePro, (1, 0, 2))
        Fimagepro = apodize(napodize, imagePro)
        
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.title(f"imagePro_{imagePro.shape}")
        plt.imshow(imagePro, cmap='gray')
        plt.axis('off')

        plt.subplot(1, 2, 2)
        plt.title(f"Fimagepro_{Fimagepro.shape}")
        plt.imshow(abs(Fimagepro), cmap='gray')
        plt.axis('off')

        plt.show()
    
        bandsDir = np.dot(sepMatrix, Fimagepro.reshape((nphases, Ny * Nx)))
        bandsDir = np.reshape(bandsDir, (nphases, Nx, Ny))

        fitorder0 = 0
        fitorder1 = 1
        overlap0, overlap1 = makeoverlaps(bandsDir, Ny, Nx, fitorder0, fitorder1, k0[d, 0], k0[d, 1], dxy, 0, OTF, lamda)
       
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.title(f"overlap0_{overlap0.shape}")
        plt.imshow(normalize(abs(overlap0)), cmap='gray')
        plt.axis('off')

        plt.subplot(1, 2, 2)
        plt.title(f"overlap1_{overlap1.shape}")
        plt.imshow(normalize(abs(overlap1)), cmap='gray')
        plt.axis('off')

        plt.show()
        new_k0, cur_modamp = fitk0andmodamps(bandsDir, overlap0, overlap1, Ny, Nx, 1, k0[d, :], dxy, 0, OTF, lamda, parameters)
        cur_k0[d, :] = new_k0

        modamp.append(cur_modamp)
    return cur_k0, modamp

if __name__ == '__main__':
    
    
   test_cal_modamp(
    128, 128, 0.488, 62.6e-3, 3, 3
)