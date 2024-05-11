# PCD_UTS_202231038_2024_ITPLN
```python
import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread("imgnama.jpg")
plt.imshow(img)
rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
plt.imshow(rgb)
```
Kode di atas menggunakan OpenCV untuk membaca gambar dari file "imgnama.jpg", kemudian menampilkannya menggunakan Matplotlib. Kemudian, kode tersebut mengubah ruang warna gambar dari BGR (Blue-Green-Red) menjadi RGB (Red-Green-Blue) untuk menampilkan gambar dengan benar dalam warna aslinya.

```python
(baris, kolom)= rgb.shape[:2]
beta = 30 #bias untuk kecerahan
citra_cerah = np.zeros((baris, kolom, 3)) #np zeros = mengubah semua elemen array menjadi 0
 
for x in range(baris) :
    for y in range(kolom) :
        gyx = rgb[x,y] + beta
        citra_cerah[x,y] = gyx
citra_cerah = citra_cerah.astype(np.uint8)
 
fig, axs = plt.subplots(2,2, figsize=(15,5))
axs[0,0].imshow(rgb)
axs[0,1].hist(img.ravel(),256,[0,256])
axs[1,0].imshow(citra_cerah)
axs[1,1].hist(citra_cerah.ravel(),256,[0,256])
plt.show()
```
Program di atas mengambil gambar yang telah dibaca sebelumnya dalam format RGB. Kemudian, program menambahkan nilai kecerahan (dalam hal ini disebut "beta") ke setiap piksel gambar, dengan memindai setiap piksel dalam gambar, menambahkan nilai beta, dan menyimpannya dalam gambar hasil. Hasilnya adalah gambar yang lebih cerah.

Program ini juga menampilkan histogram gambar asli dan gambar yang dicerahkan untuk membandingkan distribusi intensitas piksel sebelum dan sesudah peningkatan kecerahan.

```python
plt.subplot(2, 2, 1)
plt.imshow(rgb)
plt.title('Original Image')

plt.subplot(2, 2, 2)
plt.imshow(rgb[:,:,0], cmap="gray")
plt.title('Red Channel')

plt.subplot(2, 2, 3)
plt.imshow(rgb[:,:,1], cmap="gray")
plt.title('Green Channel')

plt.subplot(2, 2, 4)
plt.imshow(rgb[:,:,2], cmap="gray")
plt.title('Blue Channel')

plt.show()
```
Program di atas menggunakan Matplotlib untuk membagi gambar RGB menjadi saluran warna merah, hijau, dan biru, serta menampilkan masing-masing saluran warna dalam subplot terpisah. Ini membantu dalam memahami kontribusi setiap saluran warna terhadap gambar asli. Subplot 2x2 digunakan untuk menampilkan gambar asli bersama dengan tiga saluran warna yang dipisahkan (merah, hijau, dan biru).

```python
merah=citra_cerah[:,:,0]
fig, axs = plt.subplots(1,2, figsize = (15,5))
hist = cv2.calcHist([merah],[0],None,[256],[0,256])
axs[0].imshow(merah, cmap='gray')
axs[1].plot(hist)
plt.show()
```
Program di atas membagi gambar yang telah diubah kecerahannya menjadi saluran warna merah saja, kemudian menghitung histogram dari saluran warna merah tersebut menggunakan fungsi cv2.calcHist. Histogram kemudian ditampilkan bersama dengan gambar saluran warna merah dalam subplot. Ini membantu dalam memahami distribusi intensitas piksel pada saluran warna merah dari gambar yang telah diubah kecerahannya.

```python
hijau=citra_cerah[:,:,1] 
fig, axs = plt.subplots(1,2, figsize = (15,5))
hist = cv2.calcHist([hijau],[0],None,[256],[0,256])
axs[0].imshow(hijau, cmap='gray')
axs[1].plot(hist)
plt.show()
```
Program di atas membagi gambar yang telah diubah kecerahannya menjadi saluran warna hijau saja, kemudian menghitung histogram dari saluran warna merah tersebut menggunakan fungsi cv2.calcHist. Histogram kemudian ditampilkan bersama dengan gambar saluran warna merah dalam subplot. Ini membantu dalam memahami distribusi intensitas piksel pada saluran warna merah dari gambar yang telah diubah kecerahannya.

```python
biru=citra_cerah[:,:,2] 
fig, axs = plt.subplots(1,2, figsize = (15,5))
hist = cv2.calcHist([biru],[0],None,[256],[0,256])
axs[0].imshow(biru, cmap='gray')
axs[1].plot(hist)
plt.show()
```
Program di atas membagi gambar yang telah diubah kecerahannya menjadi saluran warna biru saja, kemudian menghitung histogram dari saluran warna merah tersebut menggunakan fungsi cv2.calcHist. Histogram kemudian ditampilkan bersama dengan gambar saluran warna merah dalam subplot. Ini membantu dalam memahami distribusi intensitas piksel pada saluran warna merah dari gambar yang telah diubah kecerahannya.

```python
gray = cv2.cvtColor(citra_cerah, cv2.COLOR_RGB2GRAY)
fig, axs = plt.subplots (2, 2, figsize=(10,10))

(thresh, binary1) = cv2.threshold(gray, 0, 0, cv2.THRESH_BINARY)
axs[0,0].imshow(binary1, cmap = 'gray')
axs[0,0].set_title('NONE')

(thresh, binary2) = cv2.threshold(gray, 105,255, cv2.THRESH_BINARY)
axs[0,1].imshow(binary2, cmap = 'binary')
axs[0,1].set_title('BLUE')

(thresh, binary3) = cv2.threshold(gray, 122, 255, cv2.THRESH_BINARY)
axs[1,0].imshow(binary3, cmap = 'binary')
axs[1,0].set_title('RED-BLUE')

(thresh, binary4) = cv2.threshold(gray, 147, 255, cv2.THRESH_BINARY)
axs[1,1].imshow(binary4, cmap = 'binary')
axs[1,1].set_title('RED-GREEN-BLUE')
```
Program di atas mengubah citra yang telah diubah kecerahannya menjadi citra keabuan (grayscale) menggunakan fungsi cv2.cvtColor. Kemudian, dilakukan proses thresholding pada citra keabuan tersebut menggunakan beberapa nilai ambang yang berbeda. Setelah itu, citra hasil thresholding ditampilkan dalam subplot bersama dengan judul yang sesuai untuk setiap subplot. Thresholding digunakan untuk menghasilkan citra biner di mana piksel yang nilainya di atas ambang tertentu menjadi putih, sedangkan piksel di bawah ambang tersebut menjadi hitam. Program ini membantu dalam memahami efek pengaturan ambang pada hasil thresholding citra keabuan.

teori pendukung https://docs.opencv.org/4.x/ <br>
https://www.geeksforgeeks.org/
