from PIL import Image
import glob
import pylab
import numpy
from sklearn.decomposition import PCA
from numpy import *
import math

def PSNR(imageA,imageB):
    MSE=sum((imageA.astype("float")-imageB.astype("float"))**2)/float(imageA.shape[0] * imageA.shape[1])
    return 20*math.log10(255/(MSE**0.5))


#input hw01-test and save it as vector
test=("hw01-test.tif")
testimage=numpy.array(Image.open(test))
testimage=testimage.reshape(1,16384)

#input training.db and save it as matrix
imlist=glob.glob("training.db/*.tif")
im = numpy.array(Image.open(imlist[0])) #open one image to get the size
m,n = im.shape #get the size of the images
imnbr = len(imlist) #get the number of image
immatrix = numpy.array([numpy.array(Image.open(imlist[i])).flatten() for i in range(imnbr)],'f')


#get meanface
meanface=immatrix.mean(axis=0)

#show the meanface
pylab.figure()
pylab.gray()
pylab.imshow(meanface.reshape(m,n))
pylab.savefig('meanface.png', dpi=80)


#every dataset-meanface 
for i in range(imnbr):
    immatrix[i]=immatrix[i]-meanface

#testface -meanface
testimage=testimage-meanface


#Keep only first K (K=5,10,15,20, and 25) coefficients and use them to reconstruct the image in the pixel domain
print"print each reconstruct face and oringinal face 's PSNR"
k=[5,10,15,20,25]
for i in k:
    pca=PCA(n_components=i)#PCA 
    pca.fit(immatrix)
    result=pca.transform(testimage)
    reconstruct=dot(result,pca.components_)#reconstruct by each eigenface coefficient dot eigenface 
    pylab.figure()
    pylab.gray()
    pylab.imshow((reconstruct+meanface).reshape(m,n))
    pylab.savefig('reconstruct_by_k='+str(i)+'.png', dpi=80)
    print PSNR(testimage+meanface,reconstruct+meanface)#print PSNR

print "############################################################################"
print "print top 10 eigenface coefficient"

pca=PCA(n_components=10)#PCA 
pca.fit(immatrix)
result=pca.transform(testimage)
print result #print top 10 eigenface coefficient


print "############################################################################"
print "top five eigenface and thier eigenvalue"
#show the top five eigenface and thier eigenvalue
for i in range(5):
    print(pca.explained_variance_ratio_[i])
    pylab.figure()
    pylab.gray()
    pylab.imshow(pca.components_[i].reshape(m,n))
    pylab.savefig('top'+str(i)+'_eigenface.png', dpi=80)    
    


pylab.show()



