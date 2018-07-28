import numpy as np 
import pandas as pd
np.random.seed(900)

def cost_function(ans ,y):
    
    
    return(np.sum((ans-y)*(ans-y)))

def sigmoid(z):
    return(1/(1+np.exp(z)))



def initialize_theta(theta,tsize,a,xn,yn):
    for i in range(tsize):
        if(i==0):
            theta.append((2*np.random.random_sample(size=(xn+1,a[0])))-1)
        elif(i==(tsize-1)):
            theta.append((2*np.random.random_sample(size=(a[i-1]+1,yn)))-1)
        else:
            theta.append((2*np.random.random_sample(size=(a[i-1]+1,a[i])))-1)
        
    return(theta)
            
def making_hidden(m,a):
    hidden = []
    for i in a :
        hidden.append(np.ones((m,i+1)))
    
    return(hidden)

def forward(x,theta,hidden):
    
    print("**********")
    
    hidden[0][:,1:]= sigmoid(np.dot(x,theta[0]))
    print(0)
    print(hidden[0])
    for i in range(1,len(hidden)):
        print(i)
        print(sigmoid(np.dot(hidden[i-1],theta[i])))
        hidden[i][:,1:]= sigmoid(np.dot(hidden[i-1],theta[i]))
        print(hidden[i][:,1:])
        
    ans = sigmoid(np.dot(hidden[-1],theta[-1]))
   
   
    print("***********")
    return(hidden,ans)
        
def deltas(y , ans , theta,hidden, x):
    delta = []
    delta.append(y-ans)
    
    delta.insert(0,((np.dot(delta[0],theta[len(hidden)].T))*(hidden[-1]*(1-hidden[-1]))))
    for i in range(len(hidden)-2, -1 , -1):
        delta.insert(0,((np.dot(delta[0][:,1:],theta[i+1].T))*(hidden[i]*(1-hidden[i]))))
    
    return(delta)
def gradients(delta,hidden,x):
    gradient = []
    gradient.append(np.dot(hidden[-1].T,delta[-1]))
    for i in range(len(hidden)-2,-1,-1):
        k= delta[i+1][:,1:]
        
        gradient.insert(0,np.dot(hidden[i].T,k))
    gradient.insert(0,(np.dot(x.T,delta[0][:,1:])))
    return(gradient)
def update(x,y,m,theta,hidden,iterations, alpha):
   
    for i in range(iterations):
        hidden , ans = forward(x,theta,hidden)
        delta = deltas(y,ans,theta,hidden,x)
        gradient = gradients(delta , hidden , x)
        for j in range(len(theta)):  
            theta[j]= theta[j]-gradient[j]
        
    return(theta)
        
def main():
    data = np.loadtxt('image1.txt')
    
    x1 = data[:,:2]
    y1 = data[:,2]
    y = np.zeros((np.size(y1),2))
    for i in range(np.size(y1)):
        y[i,int(y1[i]-1)]=1
    
        
    
    
    m = np.size(x1,0) 
    xn = np.size(x1,1)
    yn = np.size(y,1)
    x = np.ones((m,xn+1))
    x[:,1:]= x1
   
    a= [1,2,3,4]
    
    
    theta =[]
    tsize = len(a)+1
    theta = initialize_theta(theta,tsize,a,xn,yn)  
    hidden = making_hidden(m,a)
    
    iterations = 100
    alpha = 0.03
    theta = update(x,y,m,theta,hidden,iterations,alpha)
    
    
    
    

    
    
    hidden , ans = forward(x, theta, hidden)
    
    q = 0 
    q1 = 0 
    for i in range(np.size(ans,0)):
        t = 1
        for j in range(2):
            if(np.round(ans[i][j])!=y[i][j]):
                t = 0
                 
        if(t==1):
            q+=1
        q1+=1
    
    
    
    
    
main()