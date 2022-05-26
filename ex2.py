import nmf
import numpy as np

k=2
V = np.array([
[1, 1],
[2, 1],
[3, 1.2],
[4, 1],
[5, 0.8],
[6, 1]
])

W_sk, H_sk, f_sk = nmf.NMF_sk(V,k)

tol=f_sk
W, H, f = nmf.NMF(V,k,tol, steps_W=5, print_=True)

print("V:\n", V, V.shape)
print("W_sk:\n", W_sk, W_sk.shape)
print("H_sk:\n", H_sk, H_sk.shape)
print("|V-WH|_sk:\n", f_sk)
print()
print("W:\n", W, W.shape)
print("H:\n", H, H.shape)
print("|V-WH|:", f)
