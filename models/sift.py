import cv2
import numpy as np

class ImageFeatureMatcher:
    def __init__(self):
        self.sift = cv2.SIFT_create(nfeatures=200)
        self.bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
        
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)  # 指定检查次数，越大精度越高
        self.flann = cv2.FlannBasedMatcher(index_params, search_params)
        
        self.feature_library = {}  # Stores image descriptors and keypoints

    def add_to_library(self, img_id, img):
        """Extract features from an image and store them in the library."""
        keypoints, descriptors = self.sift.detectAndCompute(img, None)
        if descriptors is not None:
            self.feature_library[img_id] = {
                'keypoints': keypoints,
                'descriptors': descriptors
            }
        else:
            print(f"Failed to extract descriptors for image {img_id}")

    def compute_similarity_bf(self, img1_keypoints, img1_descriptors, img2_keypoints, img2_descriptors):
        matches = self.bf.match(img1_descriptors, img2_descriptors)
        
        # 提前过滤掉距离较大的匹配点
        matches = sorted(matches, key=lambda x: x.distance)
        
        if len(matches) < 4:
            return float('inf')
        
        src_pts = np.float32([img1_keypoints[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([img2_keypoints[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
        
        H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        if mask is None or np.count_nonzero(mask) < 4:
            return float('inf')
        
        # 计算内点匹配距离的平均值
        inlier_distances = np.array([matches[i].distance for i in range(len(matches)) if mask[i]])
        similarity_score = np.mean(inlier_distances)
        
        return similarity_score

    def compute_similarity_flann(self, img1_keypoints, img1_descriptors, img2_keypoints, img2_descriptors):
        # 使用FLANN的knnMatch找到每个描述符的两个最近邻
        matches = self.flann.knnMatch(img1_descriptors, img2_descriptors, k=2)
        # 过滤匹配对，进行Lowe比值测试
        good_matches = []
        for m, n in matches:
            if m.distance < 0.7 * n.distance:  # 比值测试的阈值可以调整
                good_matches.append(m)
        
        # 限制最多使用50个最优匹配点
        good_matches = sorted(good_matches, key=lambda x: x.distance)[:50]
        
        # 如果匹配点少于4个，则无法计算单应性矩阵
        if len(good_matches) < 4:
            return float('inf')
        
        # 获取匹配点的坐标
        src_pts = np.float32([img1_keypoints[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([img2_keypoints[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        
        # 使用RANSAC估计单应性矩阵
        H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        if mask is None or np.count_nonzero(mask) < 4:
            return float('inf')
        
        # 计算内点匹配距离的平均值作为相似度得分
        inlier_distances = np.array([good_matches[i].distance for i in range(len(good_matches)) if mask[i]])
        similarity_score = np.mean(inlier_distances)
        
        return similarity_score


    def find_top_k_similar(self, img, k=1):
        """Find the top k most similar images in the library to the provided image."""
        keypoints, descriptors = self.sift.detectAndCompute(img, None)
        if descriptors is None:
            print("Failed to extract descriptors for the input image.")
            return []

        # Store similarity scores for each image in the library
        scores = []
        for img_id, data in self.feature_library.items():
            score = self.compute_similarity_flann(
                keypoints, descriptors, 
                data['keypoints'], data['descriptors']
            )
            scores.append((img_id, score))

        # Sort by similarity score (ascending order) and return the top k matches
        scores = sorted(scores, key=lambda x: x[1])
        return scores[:k]


if __name__ == "__main__":
    matcher = ImageFeatureMatcher()
    img_path = 'datasets/20241106/1/1d57083d-0546-487c-9587-7762865ca505.png'
    img_name = '1d57083d-0546-487c-9587-7762865ca505.png'
    matcher.add_to_library(img_name, cv2.imread(img_path))

    img_path = 'datasets/20241106/2/08c7690d-b6e4-46ca-9aca-7ee8efce8d84.png'
    img_name = '08c7690d-b6e4-46ca-9aca-7ee8efce8d84.png'
    matcher.add_to_library(img_name, cv2.imread(img_path))
    
    img_path = 'datasets/20241106/3/driver_license_font_fake_2.png'
    img_name = 'driver_license_font_fake_2.png'
    matcher.add_to_library(img_name, cv2.imread(img_path))
    
    import os
    # import pdb;pdb.set_trace()
    # img_path = 'datasets/20241106/1'
    img_path = 'datasets/real'
    import time
    start = time.time()
    for img in os.listdir(img_path):
        img_file = cv2.imread(os.path.join(img_path, img))
        top_k_matches = matcher.find_top_k_similar(img_file, k=1)
        for img_id, score in top_k_matches:
            # if score < 100:
            print(f"{img} : {img_id} : {score}")
    print(time.time() - start)
    print((time.time() - start)/len(os.listdir(img_path)))
    
    
