import cv2
import numpy as np

class ImageSimilarity:
    def __init__(self):
        # 初始化SIFT特征提取器
        self.sift = cv2.SIFT_create()
    
    def extract_features(self, image):
        """
        提取SIFT特征点和描述子
        """
        keypoints, descriptors = self.sift.detectAndCompute(image, None)
        return keypoints, descriptors
    
    def match_features(self, descriptors1, descriptors2):
        """
        使用KNN匹配特征点并应用比值测试
        """
        bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
        matches = bf.knnMatch(descriptors1, descriptors2, k=2)
        good_matches = []
        for m, n in matches:
            if m.distance < 0.75 * n.distance:  # Lowe's ratio test
                good_matches.append(m)
        return good_matches
    
    def compute_ransac_inliers(self, keypoints1, keypoints2, matches):
        """
        使用RANSAC估计单应性矩阵并筛选内点
        """
        if len(matches) < 4:  # RANSAC至少需要4个点
            return [], []

        # 提取匹配点
        src_pts = np.float32([keypoints1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([keypoints2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

        # RANSAC估计单应性矩阵
        _, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        inliers = [m for i, m in enumerate(matches) if mask[i]]

        return inliers

    def compute_similarity(self, keypoints1, keypoints2, inliers):
        """
        计算内点的平均距离作为相似度分数
        """
        if not inliers:
            return float('inf')  # 如果没有内点，返回一个较大值
        
        distances = [
            np.linalg.norm(
                np.array(keypoints1[m.queryIdx].pt) - np.array(keypoints2[m.trainIdx].pt)
            )
            for m in inliers
        ]
        return np.mean(distances)

    def infer(self, image1, image2):
        """
        输入两张图像，输出相似度分数
        """
        # 转为灰度图（SIFT只支持灰度图像）
        gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY) if len(image1.shape) == 3 else image1
        gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY) if len(image2.shape) == 3 else image2

        # 提取特征点和描述子
        keypoints1, descriptors1 = self.extract_features(gray1)
        keypoints2, descriptors2 = self.extract_features(gray2)

        # 特征匹配
        matches = self.match_features(descriptors1, descriptors2)

        # 使用RANSAC筛选内点
        inliers = self.compute_ransac_inliers(keypoints1, keypoints2, matches)

        # 计算相似度分数
        score = self.compute_similarity(keypoints1, keypoints2, inliers)
        return score


# 示例使用
if __name__ == "__main__":
    # 加载两张图片
    img1 = cv2.imread("image1.jpg")
    img2 = cv2.imread("image2.jpg")

    # 创建相似度计算对象
    similarity_calculator = ImageSimilarity()

    # 计算相似度分数
    score = similarity_calculator.infer(img1, img2)
    print(f"Image Similarity Score: {score}")
