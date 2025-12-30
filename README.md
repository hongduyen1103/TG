import cv2
import numpy as np
import matplotlib.pyplot as plt

# ============================================================================
# PHẦN 1: THAO TÁC CƠ BẢN VỚI ẢNH VÀ VIDEO
# ============================================================================

# 1. Đọc ảnh
img = cv2.imread(r'image.png')

# 2. Hiện ảnh
cv2.imshow('Hienanh', img)

# 3. Lưu ảnh
if cv2.imwrite('Luuanh.png', img):
    cv2.waitKey(0) & 0xFF == ord('s')

# 4. Đọc và hiển thị video
cap = cv2.VideoCapture('video.mp4')

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    cv2.imshow('Video', frame)
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break
cap.release()

# ============================================================================
# PHẦN 2: TIỀN XỬ LÝ ẢNH - CHUYỂN ĐỔI MÀU SẮC
# ============================================================================

# 5. Chuyển đổi ảnh xám
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 6. Chuyển đổi màu BGR sang RGB
rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# 7. Hiển thị ảnh với matplotlib
plt.subplot(1,2,1)
plt.imshow(cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
plt.title('Anh goc')
plt.show()
plt.axis('off')

# ============================================================================
# PHẦN 3: BIẾN ĐỔI HÌNH HỌC
# ============================================================================

# 8. Cắt ảnh (cropping)
crop = img[0:100,  0:100]

h, w = img.shape[:2]  # lấy chiều cao và chiều rộng của ảnh

# Cắt ảnh phía trên bên phải
Crop = img[0:y, x:w]

# Cắt ảnh phía trên bên trái
Crop = img[0:h, 0:w]  # h, w nhập vào từ bàn phím

# Cắt ảnh phía dưới bên trái
Crop = img[y:h, 0:x]

# Cắt ảnh phía dưới bên phải
Crop = img[y:h, x:w]

# Cắt ảnh có tọa độ và chiều cao, chiều cao nhập vào từ bàn phím
Crop = img[y:h + y, x:w + x]

# Cắt 1 vùng ảnh với tọa độ góc trên trái, dưới phải được nhập vào
crop = img[y1:y2, x1:x2]

# 9. Thay đổi kích thước ảnh
kichthuoc = cv2.resize(img, 100 ,200)

# 10. Xoay ảnh
h, w = img.shape[:2]
xoayanh = cv2.getRotationMatrix2D(center=(h/2, w/2), angle=45, scale=1)

# 11. Dịch chuyển ảnh
M = np.float32([[1, 0, 50], [0, 1, 100]])
dichanh = cv2.warpAffine(img, M, (w, h))

# 12. Biến đổi phối cảnh
pts1 = np.float32([[], [], [], []])
pts2 = np.float32([[], [], [], []])
M = cv2.getAffineTransform(pts1, pts2)
phoicanh = cv2.warpPerspective(img, M, (w, h))

# ============================================================================
# PHẦN 4: ĐIỀU CHỈNH ẢNH
# ============================================================================

# 13. Âm bản (negative)
amban = 255 - img

# 14. Điều chỉnh độ sáng và độ tương phản
tuongphan = cv2.convertScaleAbs(img, alpha=1.5, beta=50)

# ============================================================================
# PHẦN 5: LỌC ẢNH - KHỬ NHIỄU
# ============================================================================

# 15. Lọc trung bình
blur = cv2.blur(img, (5,5))

# 16. Lọc trung vị
medianblur = cv2.medianBlur(img, 5)

# 17. Lọc Gaussian
gaussian = cv2.GaussianBlur(img, (5,5), 5)

# 18. Lọc song phương (Bilateral Filter)
bilateral = cv2.bilateralFilter(img, 5, 10, 15)

#Bộ lọc Filter2D
kernel = np.ones((5,5), np.uint8)/25
filter = cv2.filter2D(img, -1, kernel=kernel)

# ============================================================================
# PHẦN 6: PHÂN NGƯỠNG - TẠO ẢNH NHỊ PHÂN
# ============================================================================

# 19. Phân ngưỡng nhị phân cơ bản
_, thresh_binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

# 20. Phân ngưỡng thích nghi
thresh_adaptive = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)

# 21. Phân ngưỡng tối ưu Otsu
_, thresh_otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU + cv2.THRESH_BINARY)

# ============================================================================
# PHẦN 7: TÁCH BIÊN - EDGE DETECTION
# ============================================================================

# 22. Tách biên Sobel
sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
sobel = cv2.magnitude(sobelx, sobely)
sobel = cv2.convertScaleAbs(sobel)

# 23. Tách biên Canny
canny = cv2.Canny(img, 100, 200)

# 24. Tách biên Laplacian
laplican = cv2.Laplacian(gray, cv2.CV_64F, ksize=3)
laplican_result = cv2.convertScaleAbs(laplican)

# ============================================================================
# PHẦN 8: BIẾN ĐỔI HÌNH THÁI - MORPHOLOGICAL OPERATIONS
# ============================================================================

# Sử dụng ảnh nhị phân từ phân ngưỡng Otsu
thresh = thresh_otsu

# 25. Giãn ảnh (Dilate)
dilate = cv2.dilate(thresh, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3)), iterations=1)

# 26. Co ảnh (erode)
erode = cv2.erode(thresh, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3)), iterations=1)

# 27. Mở ảnh (Opening)
opending = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5)))

# 28. Đóng ảnh (Closing)
closing = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5)))

# ============================================================================
# PHẦN 9: PHÂN TÍCH ẢNH NÂNG CAO
# ============================================================================

# 29. Tìm và vẽ contours
contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
img_contours = img.copy()
cv2.drawContours(img_contours, contours, -1, (0, 255, 0), 2)

# ============================================================================
# PHẦN 10: GIAO DIỆN NGƯỜI DÙNG
# ============================================================================

# 30. Tạo Trackbar để điều khiển
import cv2

def nothing(x):
    pass

cv2.namedWindow("Image")

# tạo trackbar tên Threshold
cv2.createTrackbar("Threshold", "Image", 0, 255, nothing)

while True:
    # lấy giá trị trackbar
    thr = cv2.getTrackbarPos("Threshold", "Image")

    img = 255 * (thr % 2) * np.ones((200, 400), dtype="uint8")  # ví dụ minh hoạ
    cv2.imshow("Image", img)

    if cv2.waitKey(1) & 0xFF == 27:  # nhấn ESC để thoát
        break

cv2.destroyAllWindows()

# ============================================================================
# PHẦN 11: ĐÓNG ỨNG DỤNG
# ============================================================================

# Đợi phím bấm
cv2.waitKey(0)

# Đóng tất cả cửa sổ
cv2.destroyAllWindows()
