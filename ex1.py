import nmf
import numpy as np

k=5
V = np.array([
[2,2,2],
[1,1,1],
[1,2,2],
[1,1,2]
])

W_sk, H_sk, f_sk = nmf.NMF_sk(V,k)

tol=f_sk
W, H, f = nmf.NMF(V,k,tol,True)

print("V:\n", V, V.shape)
print("W_sk:\n", W_sk, W_sk.shape)
print("H_sk:\n", H_sk, H_sk.shape)
print("|V-WH|_sk:\n", f_sk)
print()
print("W:\n", W, W.shape)
print("H:\n", H, H.shape)
print("|V-WH|:", f)
