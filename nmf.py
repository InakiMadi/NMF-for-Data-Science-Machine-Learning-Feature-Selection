import numpy as np
from sklearn.decomposition import NMF as NMFlib

seed=0
np.random.seed(seed)

def NMF_sk(V_,k_):
    model = NMFlib(n_components=k_, init='random', random_state=seed)
    W = model.fit_transform(V_)
    H = model.components_
    f = np.linalg.norm(V_ - np.matmul(W,H) )
    return W, H, f

def NMF(V_,k_,tol_,steps_W=1,print_=False):
    # Inicialización
    #W = np.ones([V.shape[0],k])
    #H = np.ones([k,V.shape[1]])
    W = np.random.rand(V_.shape[0],k_)
    H = np.random.rand(k_,V_.shape[1])
    f = np.linalg.norm(V_ - np.matmul(W,H) )

    if print_:
        print("V:\n", V_, V_.shape)
        print("W^0:\n", W, W.shape)
        print("H^0:\n", H, H.shape)
        print("|V-WH|:", f)

    # Bucle
    while f > tol_:
        # Modificación del algoritmo: modificar H tras varios pasos de W
        for i in range(0,steps_W) :
            W = W * ( np.matmul( V_ , np.transpose(H) ) / np.matmul(W,np.matmul(H,np.transpose(H))) )
        H = H * ( np.matmul( np.transpose(W) , V_ ) / np.matmul( np.matmul(np.transpose(W),W) , H) )
        f = np.linalg.norm(V_ - np.matmul(W,H) )
        if print_:
            print("|V-WH|:", f)

    return W, H, f
