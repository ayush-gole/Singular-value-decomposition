import numpy as np 
import matplotlib.pyplot as plt
import cv2

img = cv2.imread("C:\\Users\\ayush\\OneDrive\\Desktop\\_\\Codes\\CP 4 Lab\\test_cat.png", cv2.IMREAD_GRAYSCALE)
U, W, VT = np.linalg.svd(img)

print(f"W is {W.ndim}D")
print(W)

print("original image")
print(img)
print(f"Shape(A)  : {img.shape}")
print(f"Shape(U) : {U.shape}")
print(f"Shape(W) : {W.shape}")
print(f"Shape(VT) : {VT.shape}")

perc = [ .5, .9, .99, 1.0]
img_arr = []
nv_arr = []

fig, axes = plt.subplots(2, len(perc), figsize =(16, 5), constrained_layout=True)
fig.suptitle("Singular Values for different variance threshold", fontsize=16)

for i, p in enumerate(perc):

    exp_var = (W) / np.sum(W) # explainedn variance
    cum_vare = np.cumsum(exp_var) # cumulative summation

    nv = np.argmax(cum_vare >= p) + 1 # number of W required
    nv_var = cum_vare[nv-1]

    new_W, new_U, new_VT = np.diag(W[:nv]), U[:,:nv], VT[:nv, :]
    new_img = new_U @ new_W @ new_VT

    print(f"\nsingular values threshold (in percent):    {p*100}%")
    print(f"Number of singular values used( shape of W): {nv}")
    print(f"percent of {nv}th singular values : {nv_var*100}%")
    
    img_arr.append(new_img)
    nv_arr.append(nv)

    # axes[i].plot(range(len(W[:nv])), W[:nv], 'o-', label =" linear scaled")
    axes[0,i].plot(np.log(range(len(W[:nv]))), (W[:nv]), "o-", label =" log scaled")
    axes[0,i].set_title(f"{p*100}%")
    axes[0,i].set_xlabel("Singular Value Index (i)")
    axes[0,i].set_ylabel("Singular Value (σᵢ)")
    axes[0,i].grid()
    axes[0,i].legend()

# img_arr.append(img)

# fig, axes = plt.subplots(1, len(img_arr), figsize =(16, 5), constrained_layout=True)
# fig.suptitle(f"Images for X% singular values (k singular values)", fontsize=16)

for i, im in enumerate(img_arr):
    axes[1,i].imshow(im, cmap = "gray")
    if i > len(perc)-1:
        axes[1,i].set_title(f"Original Image")
        
    else:
        axes[1,i].set_title(f"X = {perc[i]*100}% ; k ={nv_arr[i]}")


