import numpy as np
import pandas as pd
def nilmaker(nil,ar):
    for i in range(np.size(nil,1)):
        ar[nil[0][i],nil[1][i]]=0
    return(ar)
def xder(x,theta,y,nill):
    a = (np.dot(theta,np.transpose(x))-y)
    b = a*nill
    xder = np.dot(np.transpose(b),theta)
    return(xder)
    
def thetader(x,theta,y,nill):
    a = (np.dot(theta,np.transpose(x))-y)
    b = a*nill
    thetader = np.dot(b,x)
    return(thetader)
def update(x,theta,y,alpha,iterations,nill):
    for i in range(iterations):
        xder1 = xder(x,theta,y,nill)
        x = x-(1/alpha)*xder1
        thetader1 = thetader(x,theta,y,nill)
        theta = theta-(1/alpha)*thetader1
    return(x,theta)

def main():
    data = pd.read_csv('matrix.csv').values
    data = data[:,1:]
    mean = pd.read_csv('mean.csv').values
    mean = np.transpose(mean[:,2:])
    y = data-mean 
    nilindexes = np.where(data==0)
    
    nill = np.ones((671,9125))
    nill = nilmaker(nilindexes,nill)
    y = y*nill
    theta = np.random.uniform(size=(671,18))
    
    x = np.random.uniform(size=(9125,18))*5
    
    x,theta= update(x,theta,y,0.1,10,nill)
    ynew = np.dot(theta,np.transpose(x))+mean
    
    dk = pd.DataFrame(ynew)
    dk.to_csv('results.csv')
    
    dk1 = pd.DataFrame(x)
    dk1.to_csv('x.csv')
    
    dk2 = pd.DataFrame(theta)
    dk2.to_csv('theta.csv')
    print("yes")
    
      
main()
