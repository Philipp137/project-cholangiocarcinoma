import argparse
import numpy as np
from PIL import Image

from joblib import Parallel, delayed
import os
import glob

class NormalizeStaining(object):
    ''' Normalize staining appearence of H&E stained images

    Output:
        Inorm: normalized image
        H: hematoxylin image
        E: eosin image

    Reference:
        A method for normalizing histology slides for quantitative analysis. M.
        Macenko et al., ISBI 2009
    '''


    def __init__(self, Io=240, alpha=1, beta=0.15):
        self.Io = Io
        self.alpha = alpha
        self.beta = beta
        self.HERef = np.asarray([[0.5626, 0.2159],
                      [0.7201, 0.8012],
                      [0.4062, 0.5581]])

        self.maxCRef = np.asarray([1.9705, 1.0308])

    def __call__(self, PILIMAGE):
    # define height and width of image
        Io = self.Io
        alpha = self.alpha
        beta  = self.beta
        HERef = self.HERef
        maxCRef = self.maxCRef
        # reshape image

        img = np.array(PILIMAGE)
        # define height and width of image
        h, w, c = img.shape

        # reshape image
        img = img.reshape((-1, 3))

        # calculate optical density
        OD = -np.log((img.astype(np.float) + 1) / Io)

        # remove transparent pixels
        ODhat = OD[~np.any(OD < beta, axis=1)]

        # compute eigenvectors
        eigvals, eigvecs = np.linalg.eigh(np.cov(ODhat.T))

        # eigvecs *= -1

        # project on the plane spanned by the eigenvectors corresponding to the two
        # largest eigenvalues
        That = ODhat.dot(eigvecs[:, 1:3])

        phi = np.arctan2(That[:, 1], That[:, 0])

        minPhi = np.percentile(phi, alpha)
        maxPhi = np.percentile(phi, 100 - alpha)

        vMin = eigvecs[:, 1:3].dot(np.array([(np.cos(minPhi), np.sin(minPhi))]).T)
        vMax = eigvecs[:, 1:3].dot(np.array([(np.cos(maxPhi), np.sin(maxPhi))]).T)

        # a heuristic to make the vector corresponding to hematoxylin first and the
        # one corresponding to eosin second
        if vMin[0] > vMax[0]:
            HE = np.array((vMin[:, 0], vMax[:, 0])).T
        else:
            HE = np.array((vMax[:, 0], vMin[:, 0])).T

        # rows correspond to channels (RGB), columns to OD values
        Y = np.reshape(OD, (-1, 3)).T

        # determine concentrations of the individual stains
        C = np.linalg.lstsq(HE, Y, rcond=None)[0]

        # normalize stain concentrations
        maxC = np.array([np.percentile(C[0, :], 99), np.percentile(C[1, :], 99)])
        tmp = np.divide(maxC, maxCRef)
        C2 = np.divide(C, tmp[:, np.newaxis])

        # recreate the image using reference mixing matrix
        Inorm = np.multiply(Io, np.exp(-HERef.dot(C2)))
        Inorm[Inorm > 255] = 254
        Inorm = np.reshape(Inorm.T, (h, w, 3)).astype(np.uint8)

        # # unmix hematoxylin and eosin
        # H = np.multiply(Io, np.exp(np.expand_dims(-HERef[:, 0], axis=1).dot(np.expand_dims(C2[0, :], axis=0))))
        # H[H > 255] = 254
        # H = np.reshape(H.T, (h, w, 3)).astype(np.uint8)
        #
        # E = np.multiply(Io, np.exp(np.expand_dims(-HERef[:, 1], axis=1).dot(np.expand_dims(C2[1, :], axis=0))))
        # E[E > 255] = 254
        # E = np.reshape(E.T, (h, w, 3)).astype(np.uint8)

        return Image.fromarray(Inorm)




def normalizeStaining(img, saveFile=None, Io=240, alpha=1, beta=0.15):
    ''' Normalize staining appearence of H&E stained images

    Example use:
        see test.py

    Input:
        I: RGB input image
        Io: (optional) transmitted light intensity

    Output:
        Inorm: normalized image
        H: hematoxylin image
        E: eosin image

    Reference:
        A method for normalizing histology slides for quantitative analysis. M.
        Macenko et al., ISBI 2009
    '''

    HERef = np.array([[0.5626, 0.2159],
                      [0.7201, 0.8012],
                      [0.4062, 0.5581]])

    maxCRef = np.array([1.9705, 1.0308])

    # define height and width of image
    h, w, c = img.shape

    # reshape image
    img = img.reshape((-1, 3))

    # calculate optical density
    OD = -np.log((img.astype(float) + 1) / Io)

    # remove transparent pixels
    ODhat = OD[~np.any(OD < beta, axis=1)]

    # compute eigenvectors
    eigvals, eigvecs = np.linalg.eigh(np.cov(ODhat.T))

    # eigvecs *= -1

    # project on the plane spanned by the eigenvectors corresponding to the two
    # largest eigenvalues
    That = ODhat.dot(eigvecs[:, 1:3])

    phi = np.arctan2(That[:, 1], That[:, 0])

    minPhi = np.percentile(phi, alpha)
    maxPhi = np.percentile(phi, 100 - alpha)

    vMin = eigvecs[:, 1:3].dot(np.array([(np.cos(minPhi), np.sin(minPhi))]).T)
    vMax = eigvecs[:, 1:3].dot(np.array([(np.cos(maxPhi), np.sin(maxPhi))]).T)

    # a heuristic to make the vector corresponding to hematoxylin first and the
    # one corresponding to eosin second
    if vMin[0] > vMax[0]:
        HE = np.array((vMin[:, 0], vMax[:, 0])).T
    else:
        HE = np.array((vMax[:, 0], vMin[:, 0])).T

    # rows correspond to channels (RGB), columns to OD values
    Y = np.reshape(OD, (-1, 3)).T

    # determine concentrations of the individual stains
    C = np.linalg.lstsq(HE, Y, rcond=None)[0]

    # normalize stain concentrations
    maxC = np.array([np.percentile(C[0, :], 99), np.percentile(C[1, :], 99)])
    tmp = np.divide(maxC, maxCRef)
    C2 = np.divide(C, tmp[:, np.newaxis])

    # recreate the image using reference mixing matrix
    Inorm = np.multiply(Io, np.exp(-HERef.dot(C2)))
    Inorm[Inorm > 255] = 254
    Inorm = np.reshape(Inorm.T, (h, w, 3)).astype(np.uint8)

    # unmix hematoxylin and eosin
    H = np.multiply(Io, np.exp(np.expand_dims(-HERef[:, 0], axis=1).dot(np.expand_dims(C2[0, :], axis=0))))
    H[H > 255] = 254
    H = np.reshape(H.T, (h, w, 3)).astype(np.uint8)

    E = np.multiply(Io, np.exp(np.expand_dims(-HERef[:, 1], axis=1).dot(np.expand_dims(C2[1, :], axis=0))))
    E[E > 255] = 254
    E = np.reshape(E.T, (h, w, 3)).astype(np.uint8)

    if saveFile is not None:
        Image.fromarray(Inorm).save(saveFile + '.png')
        Image.fromarray(H).save(saveFile + '_H.png')
        Image.fromarray(E).save(saveFile + '_E.png')

    return Inorm, H, E

def check_file(file_path, move_dest):
            import shutil
            img = np.array(Image.open(file_path))
            h,w,c = img.shape
            if not h ==  w:
                print("shapes differ:",file_path)
            try:
                normalizeStaining(img=img,
                          Io=args.Io,
                          alpha=args.alpha,
                          beta=args.beta)
            except:
                shutil.move(file_path, move_dest) 
                print("corrupted:", file_path)


def check_parallel(data_path, destination, n_jobs=24):
        files = glob.iglob(data_path+"**/*.png",recursive=True)
        check = lambda file_path: check_file(file_path,destination)
        res = Parallel(n_jobs=n_jobs)(map(delayed(check), files))

if __name__ == '__main__':

    from torchvision import transforms, datasets
    import matplotlib.pyplot as plt
    from torch.utils.data import DataLoader
    import warnings
    warnings.filterwarnings("error")

    test_transform = False


    if test_transform == False:
        parser = argparse.ArgumentParser()
        parser.add_argument('--imageFile', type=str, default='example1.tif', help='RGB image file')
        parser.add_argument('--saveFile', type=str, default='output', help='save file')
        parser.add_argument('--Io', type=int, default=240)
        parser.add_argument('--alpha', type=float, default=1)
        parser.add_argument('--beta', type=float, default=0.15)
        args = parser.parse_args()

        #img = np.array(Image.open(args.imageFile))

        rootdir = '/work/nb671233/data/CCC_01/mainz/tiles/'
        corrupted_files_destination = '/work/nb671233/data/CCC_01/mainz/tiles/warnings/'
        check_parallel(rootdir,corrupted_files_destination,n_jobs=24)

        print("end")
        #for fname in glob.iglob(rootdir+"**/*.png",recursive=True):
        #    img = np.array(Image.open(fname))
        #    try:
        #        normalizeStaining(img=img,
        #                  Io=args.Io,
        #                  alpha=args.alpha,
        #                  beta=args.beta)
        #    except:
        #        print("corrupted:", fname)

    else:
        rootdir = '/work/nb671233/data/CCC_01/CCC/train/'
        #custom_trafo = transforms.Compose([transforms.ToTensor()])
        #dataset = datasets.ImageFolder(root=rootdir,transform=custom_trafo)
        #train_dataloader = DataLoader(dataset, batch_size=64, shuffle=False)
        #batch,labels = next(iter(train_dataloader))
        #plt.subplot(1,2,1)
        #plt.imshow(batch[0].T)

        custom_trafo = transforms.Compose([transforms.Resize(256), NormalizeStaining(), transforms.ToTensor()])
        dataset = datasets.ImageFolder(root=rootdir, transform=custom_trafo)
        train_dataloader = DataLoader(dataset, batch_size=64, shuffle=False)
        batch, labels = next(iter(train_dataloader))
        plt.subplot(1, 2, 2)
        plt.imshow(batch[0].T)

