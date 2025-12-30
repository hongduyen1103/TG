# Danh Sách Đề Tài TGMT (Xử Lý Ảnh Số)

## Mục Lục
- [Đề 1 (D1)](#đề-1-d1)
- [Đề 2 (D2)](#đề-2-d2)
- [Đề 3 (D3)](#đề-3-d3)
- [Đề 4 (D4)](#đề-4-d4)
- [Đề 5 (D5)](#đề-5-d5)

---

## Đề 1 (D1)

### Bài 1: Phân Ngưỡng Nhị Phân và Tìm Contour

**Đề Bài:**
Viết chương trình đọc ảnh, nhập vào giá trị ngưỡng và thực hiện phân ngưỡng nhị phân. Tìm và vẽ contour của ảnh. Hiển thị và lưu ảnh kết quả.

**Code:**
```python
# Bài 1. Viết chương trình đọc ảnh, nhập vào giá trị ngưỡng và thực hiện phân ngưỡng nhị phân.
# Tìm và vẽ contour của ảnh. Hiển thị và lưu ảnh kết quả.

import cv2
import numpy as np

img = cv2.imread(r'image.png')

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

x = int(input('Nhap vao gia tri nguong: '))
_, thresh_binary = cv2.threshold(gray, x, 255, cv2.THRESH_BINARY)

contours, _ = cv2.findContours(thresh_binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
gray_contours = gray.copy()
cv2.drawContours(gray_contours, contours, -1, (0, 255, 0), 2)

cv2.imshow('Hien anh', gray_contours)

if cv2.imwrite('Luuanh.png', gray_contours):
    cv2.waitKey(0) & 0xFF == ord('s')
    
cv2.destroyAllWindows()
```

---

### Bài 2: Xử Lý Video - Phân Ngưỡng Thích Nghi và Phép Co

**Đề Bài:**
Viết chương trình đọc video, xem video với màu xám. Ấn phím x để lấy ra 1 ảnh, thực hiện phép phân ngưỡng thích nghi và phép co. Lưu ảnh kết quả.

**Code:**
```python
# Bài 2. Viết chương trình đọc video, xem video với màu xám.
# Ấn phím x để lấy ra 1 ảnh, thực hiện phép phân ngưỡng thích nghi và phép co. Lưu ảnh kết quả.

import cv2
import numpy as np

cap = cv2.VideoCapture('video.mp4')

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cv2.imshow('Video', gray)
    key = cv2.waitKey(30) & 0xFF
    if key == 27:
        break
    if key == ord('x'):
        thresh_adaptive = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)
        kernel = np.ones((5,5), np.float32)/ 25
        filter = cv2.filter2D(thresh_adaptive, -1, kernel=kernel)
        erode = cv2.erode(filter, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3)), iterations=1)
        
        cv2.imshow('Ket qua', erode)
        cv2.imwrite('Luuanh.png', erode)
        
cap.release()
cv2.destroyAllWindows()
```

---

### Bài 3: Trackbar cho Bộ Lọc Gauss và Phép Mở

**Đề Bài:**
a. Đọc ảnh, tạo trackbar lấy kích thước cho kernel của bộ lọc Gauss
b. Ấn phím s để lưu ảnh. Thực hiện phép mở trên ảnh này

**Code:**
```python
# Bài 3
# a. Đọc ảnh, tạo trackbar lấy kích thước cho kernel của bộ lọc Gauss
# b. Ấn phím s để lưu ảnh. Thực hiện phép mở trên ảnh này

import cv2
import numpy as np

img = cv2.imread(r'image.png')

def nothing(x):
    pass

cv2.namedWindow('Gaussian')

cv2.createTrackbar('Kernel', 'Gaussian', 1, 31, nothing)

while True:
    k = cv2.getTrackbarPos('Kernel', 'Gaussian')
    
    if k < 1:
        k = 1
    if k % 2 == 0:
        k += 1
    
    gaussian = cv2.GaussianBlur(img, (k,k), 0)
    
    cv2.imshow('Gaussian', gaussian)
    
    key = cv2.waitKey(1) & 0xFF
    
    if key == ord('s'):
        cv2.imwrite('Gaussian.png', gaussian)
    
        opending = cv2.morphologyEx(gaussian, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5)))
        
        cv2.imshow('Anh mo', opending)
        cv2.imwrite('Luuanhmo.png', opending)
    
    if key == 27:
        break

cv2.destroyAllWindows()
```

---

### Bài 4: Xử Lý Video - Thông Tin FPS, Phân Ngưỡng Thích Nghi và Phép Mở

**Đề Bài:**
a. Đọc video, hiển thị thông tin số khung hình trong 1 giây
b. Ấn phím s để lấy ra 1 ảnh. Thực hiện phân ngưỡng thích nghi trên ảnh này. Thực hiện phép mở (opening). Lưu lại ảnh kết quả

**Code:**
```python
# Bài 4
# a. Đọc video, hiển thị thông tin số khung hình trong 1 giây
# b. Ấn phím s để lấy ra 1 ảnh. Thực hiện phân ngưỡng thích nghi trên ảnh này. Thực hiện phép mở (opening). Lưu lại ảnh kết quả

import cv2
import numpy as np

cap = cv2.VideoCapture('video.mp4')

fps = cap.get(cv2.CAP_PROP_FPS)
print("FPS của video:", fps)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    cv2.imshow('Hien video', frame)
    key = cv2.waitKey(10) & 0xFF
    if key == 27:
        break
    if key == ord('s'):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        thresh_adaptive = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)
        opending = cv2.morphologyEx(thresh_adaptive, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5)))
    
        cv2.imwrite('Luuanhmo.png', opending)
        cv2.imshow('Hienanhmo', opending)
    
cap.release()
cv2.destroyAllWindows()
```

---

## Đề 2 (D2)

### Bài 1: Xử Lý Ảnh Cơ Bản - Âm Bản, Lọc Và Phân Ngưỡng

**Đề Bài:**
1. Viết chương trình đọc ảnh và thực hiện các công việc sau:
   - In ra kích thước của ảnh
   - Biến đổi ảnh âm bản
   - Nhập vào tọa độ 1 điểm, in ra giá trị màu của ảnh ban đầu và ảnh âm bản tại tọa độ vừa nhập
   - Lọc ảnh bằng bộ lọc song phương
   - Phân ngưỡng ảnh bằng phương pháp phân ngưỡng thích nghi
   - Hiện ảnh ban đầu, ảnh âm bản, ảnh lọc, ảnh phân ngưỡng lên matplotlib

**Code:**
```python
# 1. Viết chương trình đọc ảnh và thực hiện các công việc sau:
# In ra kích thước của ảnh
# Biến đổi ảnh âm bản
# Nhập vào tọa độ 1 điểm, in ra giá trị màu của ảnh ban đầu và ảnh âm bản tại tọa độ vừa nhập
# Lọc ảnh bằng bộ lọc song phương
# Phân ngưỡng ảnh bằng phương pháp phân ngưỡng thích nghi
# Hiện ảnh ban đầu, ảnh âm bản, ảnh lọc, ảnh phân ngưỡng lên matplotlib

import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread(r'image.png')

h, w = img.shape[:2]

print('Kich thuoc cua anh la chieu rong, chieu cao: ', w, h)

amban = 255 - img

x = int(input('Nhap x:' ))
y = int(input('Nhap y: '))

print('Mau anh goc: ', img[y, x])
print('Mau anh am ban: ', amban[y, x])

bilater = cv2.bilateralFilter(img, 5, 10, 15)

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

thresh_adaptive = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)

plt.subplot(2,2,1)
plt.imshow(cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
plt.title('Anh goc')

plt.subplot(2,2,2)
plt.imshow(amban)
plt.title('Anh am ban')

plt.subplot(2,2,3)
plt.imshow(gray, cmap='gray')
plt.title('Anh xam')

plt.subplot(2,2,4)
plt.imshow(thresh_adaptive, cmap='gray')
plt.title('Anh phan nguong')

plt.show()
plt.axis('off')

cv2.waitKey(0)
cv2.destroyAllWindows()
```

---

### Bài 2: Trackbar Điều Chỉnh Độ Sáng

**Đề Bài:**
2. Tạo trackbar thực hiện thay đổi độ sáng của ảnh. Ấn phím s để lưu ảnh. Ấn q để thoát khỏi chương trình

**Code:**
```python
# 2. Tạo trackbar thực hiện thay đổi độ sáng của ảnh. Ấn phím s để lưu ảnh. Ấn q để thoát khỏi chương trình

import cv2

img = cv2.imread(r'image.png')

def nothing(x):
    pass

cv2.namedWindow('Anh')
cv2.createTrackbar('DoSang','Anh', 100, 200, nothing)

while True:
    dosang = cv2.getTrackbarPos('DoSang', 'Anh')
    
    beta = dosang -  100
    ds = cv2.convertScaleAbs(img, alpha=1, beta=beta)
    
    cv2.imshow('Anh', ds)
    
    key = cv2.waitKey(1) & 0xFF
    if key == ord('s'):
        cv2.imwrite('AnhDoSang.png', ds)
    if key == ord('q'):
        break
    
cv2.destroyAllWindows()
```

---

### Bài 3: Xử Lý Video - Vẽ Contour

**Đề Bài:**
3. Đọc video, ấn phím s để vẽ contour cho ảnh và lưu lại ảnh sau khi vẽ contour

**Code:**
```python
# 3. Đọc video, ấn phím s để vẽ contour cho ảnh và lưu lại ảnh sau khi vẽ contour

import cv2

cap = cv2.VideoCapture('video.mp4')

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    cv2.imshow('Video',frame)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    key = cv2.waitKey(30) & 0xFF
    if key == ord('s'):
        _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        gray_bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        cv2.drawContours(gray_bgr, contours, -1, (0, 255, 0), 2)
        cv2.imshow('Anh contours', gray_bgr)
        cv2.imwrite('Luuanh_Contous.png', gray_bgr)
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()
```

---

### Bài 4: Xử Lý Ảnh - Cắt Nửa Ảnh Và Tách Biên Sobel

**Đề Bài:**
4. Đọc ảnh, cắt nửa ảnh bên trái và tách biên ảnh bằng Sobel. Hiện các ảnh lên matplotlib

**Code:**
```python
# 4. Đọc ảnh, cắt nửa ảnh bên trái và tách biên ảnh bằng Sobel. Hiện các ảnh lên matplotlib

import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread(r'image.png')

h, w = img.shape[:2]

crop = img[:, :w//2]

gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
sobel = cv2.magnitude(sobelx, sobely)
sobel = cv2.convertScaleAbs(sobel)

plt.subplot(1, 3, 1)
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.title('Ảnh gốc')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.imshow(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
plt.title('Nửa trái')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.imshow(sobel, cmap='gray')
plt.title('Biên Sobel')
plt.axis('off')

plt.show()
```

---

## Đề 3 (D3)

### Bài 1: Xử Lý Ảnh - Điều Chỉnh Sáng/Tương Phản, Lọc Gauss, Laplace, Phép Mở/Đóng

**Đề Bài:**
1. Đọc ảnh và thực hiện:
   - Thay đổi độ sáng, độ tương phản với giá trị nhập vào từ bàn phím
   - Lọc ảnh bằng bộ lọc Gauss
   - Tách biên ảnh bằng Laplace
   - Thực hiện phép mở, phép đóng
   - Hiện toàn bộ các ảnh trên matplotlib

**Code:**
```python
# 1. Đọc ảnh và thực hiện:
# Thay đổi độ sáng, độ tương phản với giá trị nhập vào từ bàn phím
# - Lọc ảnh bằng bộ lọc Gauss
# - Tách biên ảnh bằng Laplace
# - Thực hiện phép mở, phép đóng
# - Hiện toàn bộ các ảnh trên matplotlib

import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread(r'image.png')

alpha = int(input('Nhap gia tri do sang: '))
beta = int(input('Nhap gia tri do tuong phan: '))
alpha_beta = cv2.convertScaleAbs(img, alpha = alpha, beta = beta)

gaussian = cv2.GaussianBlur(alpha_beta, (5,5), 15)

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

laplace = cv2.Laplacian(gray, cv2.CV_64F, ksize=3)
laplace_result = cv2.convertScaleAbs(laplace)

opending = cv2.morphologyEx(gray, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5)))
closing = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5)))

plt.subplot(2, 3, 1)
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.title('Ảnh gốc')

plt.subplot(2, 3, 2)
plt.imshow(cv2.cvtColor(alpha_beta, cv2.COLOR_BGR2RGB))
plt.title('Sáng / Tương phản')

plt.subplot(2, 3, 3)
plt.imshow(cv2.cvtColor(gaussian, cv2.COLOR_BGR2RGB))
plt.title('Lọc Gauss')

plt.subplot(2, 3, 4)
plt.imshow(laplace_result, cmap='gray')
plt.title('Biên Laplace')

plt.subplot(2, 3, 5)
plt.imshow(opending, cmap='gray')
plt.title('Phép mở')

plt.subplot(2, 3, 6)
plt.imshow(closing, cmap='gray')
plt.title('Phép đóng')

plt.axis('off')
plt.show()
```

---

### Bài 2: Trackbar Dịch Chuyển Ảnh Ngang

**Đề Bài:**
2. Đọc ảnh:
   - Tạo trackbar thực hiện dịch ảnh sang ngang

**Code:**
```python
# 2. Đọc ảnh:
# - Tạo trackbar thực hiện dịch ảnh sang ngang

import cv2
import numpy as np

img = cv2.imread(r'image.png')

def nothing(x):
    pass

cv2.namedWindow('Anh')

cv2.createTrackbar('DichAnh', 'Anh', 0, 100, nothing)
h, w = img.shape[:2]

while True:
    dx = cv2.getTrackbarPos('DichAnh', 'Anh')
    M = np.float32([[1,0,dx], [0,1,0]])
    dichanh = cv2.warpAffine(img, M, (w, h))
    cv2.imshow('Anh', dichanh)
    key = cv2.waitKey(1) & 0xFF
    if key  == 27:
        break
    
cv2.destroyAllWindows()
```

---

### Bài 3: Xử Lý Video - Lưu Ảnh, Ảnh Xám, Và Contour

**Đề Bài:**
3.Đọc video:
   - Ấn phím s để lưu ảnh.
   - Ấn phím g để biến đổi ảnh thành ảnh xám và lưu lại ảnh xám
   - Ấn phím c để vẽ contour, lưu lại ảnh sau khi vẽ

**Code:**
```python
# 3.Đọc video:
# - Ấn phím s để lưu ảnh.
# - Ấn phím g để biến đổi ảnh thành ảnh xám và lưu lại ảnh xám
# - Ấn phím c để vẽ contour, lưu lại ảnh sau khi vẽ

import cv2

cap = cv2.VideoCapture('video.mp4')

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    cv2.imshow('Video', frame)
    key = cv2.waitKey(30) & 0xFF
    if  key == ord('s'):
        cv2.imwrite('Luuanh.png', frame)
    if key == ord('g'):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        cv2.imwrite('LuuanhXam.png', gray)
    if key == ord('c'):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contour_img = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)
        cv2.drawContours(contour_img, contours, -1, (0, 255, 0), 2)
        cv2.imwrite('LuuAnh_Contous.png', contour_img)
    if key == 27:
        break
    
cap.release()
cv2.destroyAllWindows()
```

---

## Đề 4 (D4)

### Bài 1: Xử Lý Ảnh - Kích Thước, Giá Trị Màu Và Điều Chỉnh Tương Phản

**Đề Bài:**
Câu 1 (3 điểm): Đọc ảnh màu và thực hiện các yêu cầu sau:
- a. In ra chiều rộng, chiều cao của ảnh. (1.0 điểm)
- b. In giá trị màu của ảnh tại điểm có tọa độ nhập vào. (2.0 điểm)
- c. Tăng độ tương phản của ảnh với hệ số tương phản nhập vào từ bàn phím, lưu lại ảnh kết quả. (1.0 điểm)

**Code:**
```python
# Câu 1 (3 điểm): Đọc ảnh màu và thực hiện các yêu cầu sau:
# a. In ra chiều rộng, chiều cao của ảnh. (1.0 điểm)
# b. In giá trị màu của ảnh tại điểm có tọa độ nhập vào. (2.0 điểm)
# c. Tăng độ tương phản của ảnh với hệ số tương phản nhập vào từ hàn phím, lưu lại ảnh kết quả. (1.0 điểm)

import cv2

img = cv2.imread(r'image.png')

h, w = img.shape[:2]

print('Chieu rong va chieu cao cua anh: ', w, h)

x = int(input('Nhap toa do mau diem x: '))
y = int(input('Nhap toa do mau diem y: '))

print('Gia tri mau cua anh tai diem co toa do nhap vao la: ', img[y, x])

alpla = float(input('Nhap do tuong phan: '))
alpla_beta = cv2.convertScaleAbs(img, alpha=alpla, beta= 0)
cv2.imshow('Anh tuong phan nhap tu ban phim', alpla_beta)
cv2.imwrite('Luuanh.png', alpla_beta)

cv2.waitKey(0)
cv2.destroyAllWindows()
```

---

### Bài 2: Trackbar Kích Thước Kernel - Bộ Lọc Trung Bình

**Đề Bài:**
Câu 2 (2 điểm): Đọc ảnh màu và thực hiện các yêu cầu sau:
- Tạo trackbar để chọn kích thước mặt nạ lọc. (2.0 điểm)
- Lọc ảnh bằng bộ lọc trung bình với kích thước mặt nạ lấy từ trackbar. (1.0 điểm)

**Code:**
```python
# Câu 2 (2 điểm): Đọc ảnh màu và thực hiện các yêu cầu sau:
# - Tạo trackbar để chọn kích thước mặt nạ lọc. (2.0 điểm)
# - Lọc ảnh bằng bộ lọc trung bình với kích thước mặt nạ lấy từ trackbar. (1.0 điểm)

import cv2

img = cv2.imread('image.png')

def nothing(x):
    pass

cv2.namedWindow('Anh')

cv2.createTrackbar('Kernel', 'Anh', 1, 31, nothing)

while True:
    k = cv2.getTrackbarPos('Kernel', 'Anh')
    if k < 1:
        k = 0
    if k % 2 == 0:
        k += 1
    blur = cv2.blur(img, (k,k))
    cv2.imshow('Anh', blur)
    key = cv2.waitKey(1) & 0xFF
    if key == 27:
        break
cv2.destroyAllWindows()
```

---

### Bài 3: Xử Lý Ảnh - Lọc Gauss, Phân Ngưỡng Otsu, Canny Edge Detection

**Đề Bài:**
Câu 3 (3 điểm): Đọc ảnh màu và thực hiện các yêu cầu sau:
- Khử nhiễu ảnh bằng bộ lọc Gauss. (1.0 điểm)
- Thực hiện phân ngưỡng ảnh bằng thuật toán phân ngưỡng tối ưu. (1.0 điểm)
- Tách biên ảnh bằng phương pháp tách biên Canny. (0.5 điểm)
- Hiển thị ảnh ban đầu, các ảnh kết quả trên matplotlib. (0.5 điểm)

**Code:**
```python
# Câu 3 (3 điểm): Đọc ảnh màu và thực hiện các yêu cầu sau:
# - Khử nhiễu ảnh bằng bộ lọc Gauss. (1.0 điểm)
# - Thực hiện phân ngưỡng ảnh bằng thuật toán phân ngưỡng tối ưu. (1.0 điểm)
# - Tách biên ảnh bằng phương pháp tách biên Canny. (0.5 điểm)
# - Hiển thị ảnh ban đầu, các ảnh kết quả trên matplotlib. (0.5 điểm)

import cv2
import matplotlib.pyplot as plt

img = cv2.imread('image.png')

gauss = cv2.GaussianBlur(img, (5, 5), 0)

gray = cv2.cvtColor(gauss, cv2.COLOR_BGR2GRAY)

_, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

canny = cv2.Canny(gray, 100, 200)

plt.subplot(2, 2, 1)
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.title('Ảnh gốc')

plt.subplot(2, 2, 2)
plt.imshow(cv2.cvtColor(gauss, cv2.COLOR_BGR2RGB))
plt.title('Gauss')

plt.subplot(2, 2, 3)
plt.imshow(otsu, cmap='gray')
plt.title('Ngưỡng Otsu')

plt.subplot(2, 2, 4)
plt.imshow(canny, cmap='gray')
plt.title('Biên Canny')

plt.axis('off')
plt.show()
```

---

### Bài 4: Xử Lý Video - Lấy FPS Và Hiển Thị Video Xám

**Đề Bài:**
Câu 4 (2 điểm): Viết chương trình thực hiện các yêu cầu sau:
- Đọc video từ tệp, in ra thông tin số khung hình trong 1 giây của video. (1.0 điểm)
- Xem video với màu xám. (1.0 điểm)

**Code:**
```python
# Câu 4 (2 điểm): Viết chương trình thực hiện các yêu cầu sau:
# - Đọc video từ tệp, in ra thông tin số khung hình trong 1 giây của video. (1.0 điểm)
# - Xem video với màu xám. (1.0 điểm)

import cv2

cap = cv2.VideoCapture('video.mp4')

import cv2
import numpy as np

cap = cv2.VideoCapture('video.mp4')

fps = cap.get(cv2.CAP_PROP_FPS)
print("Tong so khung hinh trong 1 giay cua video: ", fps)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cv2.imshow('Video', gray)
    key = cv2.waitKey(30) & 0xFF
    if key == 27:
        break
    
cap.release()
cv2.destroyAllWindows()
```

---

## Đề 5 (D5)

### Bài 1: Xử Lý Ảnh - Kích Thước, Giá Trị Màu Tại Điểm, Cắt Ảnh

**Đề Bài:**
Câu 1 (3 điểm): Đọc ảnh màu và thực hiện các yêu cầu sau:
- a. In ra kích thước ảnh vừa đọc. (1.0 điểm)
- b. In giá trị màu của ảnh tại điểm có tọa độ (100,200). (1.0 điểm)
- c. Cắt một phần ảnh với các tọa độ nhập vào từ bàn phím, lưu ảnh vừa cắt. (1.0 điểm)

**Code:**
```python
# Câu 1 (3 điểm): Đọc ảnh màu và thực hiện các yêu cầu sau:
# a. In ra kích thước ảnh vừa đọc. (1.0 điểm)
# b. In giá trị màu của ảnh tại điểm có tọa độ (100,200). (1.0 điểm)
# c. Cắt một phần ảnh với các tọa độ nhập vào từ bàn phím, lưu ảnh vừa cắt. (1.0 điểm)

import cv2

img = cv2.imread('image.png')

h, w = img.shape[:2]
print("Kích thước ảnh:", w, h)

print("Giá trị màu BGR tại (100,200):", img[200, 100])

x1 = int(input("Nhập x1: "))
y1 = int(input("Nhập y1: "))
x2 = int(input("Nhập x2: "))
y2 = int(input("Nhập y2: "))

crop = img[y1:y2, x1:x2]
cv2.imwrite("anh_cat.png", crop)
cv2.imshow('Hienanh', crop)

cv2.waitKey(0)
cv2.destroyAllWindows()
```

---

### Bài 2: Trackbar Dịch Chuyển Ảnh (Trục X và Y)

**Đề Bài:**
Câu 2 (2 điểm): Đọc ảnh màu và thực hiện các yêu cầu sau:
- Tạo trackbar để chọn kích thước dx, dy. (1.0 điểm)
- Dịch chuyển ảnh với khoảng cách theo trục x và trục y là dx, dy lấy từ trackbar. (1.0 điểm)

**Code:**
```python
# Câu 2 (2 điểm): Đọc ảnh màu và thực hiện các yêu cầu sau:
# - Tạo trackbar để chọn kích thước dx, dy. (1.0 điểm)
# - Dịch chuyển ảnh với khoảng cách theo trục x và trục y là dx, dy lấy từ trackbar. (1.0 điểm)

import cv2
import numpy as np

img = cv2.imread('image.png')

def nothing(x):
    pass

cv2.namedWindow('Anh')

cv2.createTrackbar('dx', 'Anh', 0, 100, nothing)
cv2.createTrackbar('dy', 'Anh', 0, 100, nothing)

h, w = img.shape[:2]

while True:
    dx = cv2.getTrackbarPos('dx', 'Anh')
    dy = cv2.getTrackbarPos('dy', 'Anh')

    M = np.float32([[1, 0, dx], [0, 1, dy]])

    dichanh = cv2.warpAffine(img, M, (w, h))

    cv2.imshow('Anh', dichanh)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cv2.destroyAllWindows()
```

---

### Bài 3: Xử Lý Ảnh - Lọc Trung Bình, Phân Ngưỡng Thích Nghi, Laplace Edge Detection

**Đề Bài:**
Câu 3 (3 điểm): Đọc ảnh màu và thực hiện các yêu cầu sau:
- Khử nhiễu ảnh bằng bộ lọc trung bình. (1.0 điểm)
- Thực hiện phân ngưỡng ảnh bằng thuật toán phân ngưỡng thích nghi. (1.0 điểm)
- Tách biên ảnh bằng phương pháp tách biên Laplace. (0.5 điểm)
- Hiển thị ảnh ban đầu, các ảnh kết quả trên matplotlib. (0.5 điểm)

**Code:**
```python
# Câu 3 (3 điểm): Đọc  màu và thực hiện các yêu cầu sau:
# - Khử nhiễu ảnh bằng bộ lọc trung bình. (1.0 điểm)
# - Thực hiện phân ngưỡng ảnh bằng thuật toán phân ngưỡng thích nghi. (1.0 điểm)
# - Tách biên ảnh bằng phương pháp tách biên Laplace. (0.5 điểm)
# - Hiển thị ảnh ban đầu, các ảnh kết quả trên matplotlib. (0.5 điểm)

import cv2
import matplotlib.pyplot as plt

img = cv2.imread('image.png')

blur = cv2.blur(img, (5, 5))

gray = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)

th_adap = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)

lap = cv2.Laplacian(gray, cv2.CV_64F, ksize=3)
lap_abs = cv2.convertScaleAbs(lap)

plt.subplot(2, 2, 1)
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.title('Ảnh gốc')
plt.axis('off')

plt.subplot(2, 2, 2)
plt.imshow(cv2.cvtColor(blur, cv2.COLOR_BGR2RGB))
plt.title('Lọc trung bình')
plt.axis('off')

plt.subplot(2, 2, 3)
plt.imshow(th_adap, cmap='gray')
plt.title('Ngưỡng thích nghi')
plt.axis('off')

plt.subplot(2, 2, 4)
plt.imshow(lap_abs, cmap='gray')
plt.title('Biên Laplace')
plt.axis('off')

plt.tight_layout()
plt.show()
```

---

### Bài 4: Xử Lý Video - Lấy Ảnh Từ Video Và Phép Giãn Nở

**Đề Bài:**
Câu 4 (2 điểm): Viết chương trình thực hiện các yêu cầu sau:
- Đọc video từ tệp, lấy ra 1 ảnh từ video khi ấn phím x. (1.0 điểm)
- Thực hiện phép giãn nở (dilate) đối với ảnh cắt được. (1.0 điểm)

**Code:**
```python
# Câu 4 (2 điểm): Viết chương trình thực hiện các yêu cầu sau:
# - Đọc video từ tệp, lấy ra 1 ảnh từ video khi ấn phím x. (1.0 điểm)
# - Thực hiện phép giãn nở (dilate) đối với ảnh cắt được. (1.0 điểm)

import cv2
import numpy as np

cap = cv2.VideoCapture('video.mp4')

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    cv2.imshow('Video', frame)

    key = cv2.waitKey(30) & 0xFF

    if key == ord('x'):
        cv2.imwrite('frame.png', frame)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        kernel = np.ones((5,5), np.uint8)
        
        dilate = cv2.dilate(thresh, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3)), iterations=1)

        cv2.imwrite('frame_dilate.png', dilate)
        cv2.imshow('Dilate', dilate)

    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()
```

---

## Tóm Tắt Các Hàm Chính

### Xử Lý Ảnh Cơ Bản
- `cv2.imread()` - Đọc ảnh
- `cv2.imwrite()` - Lưu ảnh
- `cv2.imshow()` - Hiển thị ảnh
- `cv2.cvtColor()` - Chuyển đổi không gian màu
- `cv2.destroyAllWindows()` - Đóng cửa sổ

### Xử Lý Video
- `cv2.VideoCapture()` - Mở video
- `cap.read()` - Đọc khung hình
- `cap.get(cv2.CAP_PROP_FPS)` - Lấy FPS

### Phân Ngưỡng
- `cv2.threshold()` - Phân ngưỡng nhị phân
- `cv2.adaptiveThreshold()` - Phân ngưỡng thích nghi

### Lọc
- `cv2.GaussianBlur()` - Bộ lọc Gauss
- `cv2.blur()` - Bộ lọc trung bình
- `cv2.bilateralFilter()` - Bộ lọc song phương

### Tách Biên
- `cv2.Sobel()` - Sobel
- `cv2.Laplacian()` - Laplace
- `cv2.Canny()` - Canny

### Morphological Ops
- `cv2.erode()` - Phép co
- `cv2.dilate()` - Phép giãn
- `cv2.morphologyEx()` - Phép mở/đóng

### Contour
- `cv2.findContours()` - Tìm contour
- `cv2.drawContours()` - Vẽ contour

### Biến Đổi Hình Học
- `cv2.warpAffine()` - Dịch chuyển ảnh
- `cv2.convertScaleAbs()` - Điều chỉnh sáng/tương phản

### Trackbar
- `cv2.createTrackbar()` - Tạo thanh trượt
- `cv2.getTrackbarPos()` - Lấy giá trị thanh trượt

---

**Cập nhật:** Tháng 12, 2025
