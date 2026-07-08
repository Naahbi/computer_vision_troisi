
import numpy as np
from skimage.filters import frangi
import cv2
from scipy.ndimage import gaussian_filter1d
import scipy.signal as signal
import pywt
from skimage.feature import local_binary_pattern
from skimage.feature import graycomatrix, graycoprops
import joblib
import segmentation_models_pytorch as smp
import torch
import torchvision.transforms as transforms

UNET_PATH = "models/unet_weights.pth"
KMEAN_MODEL_PATH = "models/kmeans_model.joblib"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
transform_to_tensor = transforms.ToTensor()


# loading U-NET
def load_unet():
    unet = smp.Unet(
        encoder_name="efficientnet-b7",
        encoder_weights=None,
        in_channels=3,
        classes=1,
        activation=None
    )
    unet.load_state_dic(torch.load(UNET_PATH, map_location=torch.device('cpu')))
    unet.to(device)
    unet.eval()
    return unet


# Returns the Kmean model to build haar histograms
def load_kmean_model():
    return joblib.load(KMEAN_MODEL_PATH)


# Returns the svm model given the related path
def load_svm_path(svm_path):
    return joblib.load(svm_path)


# Detects body artifacts and inpaint them
def body_hair_removal(img_hsv_255, img_bgr_255):
    frangi_mask = frangi(img_hsv_255[:, :, 2], sigmas=[1, 2], gamma=5)
    frangi_normalized = cv2.normalize(
        frangi_mask, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX
    ).astype(np.uint8)

    mask = cv2.threshold(frangi_normalized, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    mask = cv2.GaussianBlur(mask, (7, 7), sigmaX=2)
    inpainted_pic = cv2.inpaint(img_bgr_255, mask, 3, cv2.INPAINT_TELEA)
    return inpainted_pic, mask


# Transrotates the given image according to its momentum
def get_transrotated_img(img_bgr_255, mask):
    def get_transrotation_matrix(mask):
        M = cv2.moments(mask)
        if M["m00"] == 0:
            return np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)  # identity matrix
        cX = int(M["m10"] / M['m00'])
        cY = int(M["m01"] / M['m00'])

        theta = 0.5 * np.arctan2(2 * M['mu11'], (M['mu20'] - M['mu02']))
        angle_degrees = np.degrees(theta)

        h, w = mask.shape[:2]
        img_cX = int(w / 2)
        img_cY = int(h / 2)

        M_combined = cv2.getRotationMatrix2D(center=(cX, cY), angle=angle_degrees, scale=1.0)
        M_combined[0, 2] += (img_cX - cX)
        M_combined[1, 2] += (img_cY - cY)

        return M_combined

    M = get_transrotation_matrix(mask)
    aligned_mask = cv2.warpAffine(mask, M, (mask.shape[1], mask.shape[0]), flags=cv2.INTER_NEAREST,
                                  borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    aligned_img = cv2.warpAffine(img_bgr_255, M, (img_bgr_255.shape[1], img_bgr_255.shape[0]), flags=cv2.INTER_LINEAR,
                                 borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))

    segmented_img = aligned_img.copy()
    segmented_img[aligned_mask == 0] = [0, 0, 0]
    return segmented_img, aligned_mask


# Returns shape asymmetry index in range [0; 1]
def shape_asymmetry_index(mask):
    binary_mask = (mask > 0).astype(np.uint8)
    mask_flip_h = cv2.flip(binary_mask, 1)
    mask_flip_v = cv2.flip(binary_mask, 0)

    diff_h = cv2.bitwise_xor(binary_mask, mask_flip_h)
    diff_v = cv2.bitwise_xor(binary_mask, mask_flip_v)

    area = np.sum(binary_mask)
    asym_area_h = np.sum(diff_h) / 2
    asym_area_v = np.sum(diff_v) / 2

    asym_index = ((asym_area_h + asym_area_v) / area)

    return asym_index / 2


# Returns color asymmetry index in range [0; 1]
def color_asymmetry_index(aligned_img_bgr, aligned_mask):
    img_lab = cv2.cvtColor(aligned_img_bgr, cv2.COLOR_BGR2Lab)
    img_lab = img_lab.astype(np.float32)

    img_flip_h = cv2.flip(img_lab, 1)
    img_flip_v = cv2.flip(img_lab, 0)

    mask_flip_h = cv2.flip(aligned_mask, 1)
    mask_flip_v = cv2.flip(aligned_mask, 0)

    diff_h_sq = (img_lab - img_flip_h) ** 2
    dist_h = np.sqrt(diff_h_sq[:, :, 0] + diff_h_sq[:, :, 1] + diff_h_sq[:, :, 2])

    diff_v_sq = (img_lab - img_flip_v) ** 2
    dist_v = np.sqrt(diff_v_sq[:, :, 0] + diff_v_sq[:, :, 1] + diff_v_sq[:, :, 2])

    valid_pixels_h = (aligned_mask > 0) & (mask_flip_h > 0)
    valid_pixels_v = (aligned_mask > 0) & (mask_flip_v > 0)

    color_asym_h = np.mean(dist_h[valid_pixels_h]) if np.sum(valid_pixels_h) > 0 else 0
    color_asym_v = np.mean(dist_v[valid_pixels_v]) if np.sum(valid_pixels_v) > 0 else 0

    total_color_asym = np.sqrt((color_asym_h ** 2 + color_asym_v ** 2) / 2)

    k = 0.05
    normalized_asym = 2 / (1 + np.exp(-k * total_color_asym)) - 1

    return normalized_asym


# Returns the border irregularity index in a range [0, 8]
def border_irregularity_index(mask, center):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    x_c = []
    y_c = []
    if len(contours) == 0:
        return None
    contour = contours[0].reshape(-1, 2)
    for pt in contour:
        x_c.append(pt[0])
        y_c.append(pt[1])

    contour = contours[0].reshape(-1, 2)

    distances = []
    angles = []

    for pt in contour:
        x, y = pt[0], pt[1]
        dist = np.sqrt((x - center[0]) ** 2 + (y - center[1]) ** 2)
        angle = np.degrees(np.arctan2(y - center[1], x - center[0]))
        distances.append(dist)
        if angle < 0:
            angle += 360
        angles.append(angle)

    distances = np.array(distances)
    angles = np.array(angles)

    irregular_ottants = 0

    sigma = max(1, np.std(distances))
    for i in range(8):
        start_angle = i * 45
        end_angle = (i + 1) * 45

        indices = np.where((angles >= start_angle) & (angles < end_angle))
        if len(indices[0]) == 0:
            continue
        octant_signal = distances[indices]

        x, y = [], []
        for idx, d in enumerate(octant_signal):
            x.append(idx)
            y.append(d)

        smoothed_signal = gaussian_filter1d(octant_signal, sigma)

        diff_signal = octant_signal - smoothed_signal

        noise_threshold = max(2.0, np.std(diff_signal) * 1.5)
        min_peak_distance = max(15, int(len(octant_signal) * 0.02))  # distance between two peaks

        peaks_protrusion, _ = signal.find_peaks(diff_signal, height=noise_threshold, distance=min_peak_distance)
        peaks_indentations, _ = signal.find_peaks(-diff_signal, height=noise_threshold, distance=min_peak_distance)

        n_irregularities = len(peaks_protrusion) + len(peaks_indentations)
        if n_irregularities > 0:
            irregular_ottants += 1

    return irregular_ottants


def extract_patches(img_gray, mask, M=10, K=24):
    y_idx, x_idx = np.where(mask > 0)
    if len(y_idx) == 0:
        return np.array([])

    y_min, y_max = y_idx.min(), y_idx.max()
    x_min, x_max = x_idx.min(), x_idx.max()

    vectors = []
    for y in range(y_min + K // 2, y_max - K // 2, M):
        for x in range(x_min + K // 2, x_max - K // 2, M):
            if mask[y, x] > 0:
                patch = img_gray[y - K // 2:y + K // 2, x - K // 2:x + K // 2]
                coeffs = pywt.wavedec2(patch, 'haar', level=3)

                vector = []
                sub_bands = [coeffs[0]] + list(coeffs[1]) + list(coeffs[2]) + list(coeffs[3])

                for sb in sub_bands:
                    vector.append(np.mean(sb))
                    vector.append(np.std(sb))

                vectors.append(vector)
    return np.array(vectors)


# Returns the haard histogram built with BoW strategy
def build_haar_histogram(vectors, kmeans_model, L=200):
    if len(vectors) == 0:
        return np.zeros(L)
    labels = kmeans_model.predict(vectors)
    hist, _ = np.histogram(labels, bins=range(L + 1), density=False)
    return hist


# Returns a vector containing the LBP histograms obtained from the given image
def extract_lbp_features(img_gray, mask):
    lbp_16 = local_binary_pattern(img_gray, P=16, R=2, method='ror')
    lbp_24 = local_binary_pattern(img_gray, P=24, R=3, method='ror')

    roi_lbp_16 = lbp_16[mask > 0]
    roi_lbp_24 = lbp_24[mask > 0]

    hist_16, _ = np.histogram(roi_lbp_16, bins=range(18), density=False)
    hist_24, _ = np.histogram(roi_lbp_24, bins=range(32), density=False)

    return np.concatenate((hist_16, hist_24))


# Returns an array containing the concatenated color histogram of the given image
def extract_color_features(img_bgr, mask, P_bins=32):
    img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    img_lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)

    channels_info = [
        (img_hsv[:, :, 0], 180), (img_hsv[:, :, 1], 256), (img_hsv[:, :, 2], 256),
        (img_lab[:, :, 0], 256), (img_lab[:, :, 1], 256), (img_lab[:, :, 2], 256)]

    color_vector = []
    for ch, upper_bound in channels_info:
        roi_pixels = ch[mask > 0]
        if len(roi_pixels) == 0:
            color_vector.extend(np.zeros(P_bins + 2))
            continue
        hist, _ = np.histogram(roi_pixels, bins=P_bins, range=(0, upper_bound), density=False)
        p = hist / (np.sum(hist) + 1e-6)
        entropy = -np.sum(p * np.log2(p + 1e-6))
        std_dev = np.std(hist)

        color_vector.extend(hist)
        color_vector.append(std_dev)
        color_vector.append(entropy)
    return np.array(color_vector)


# Returns an array containing the GLCM feats extracted from the given image
def extract_glcm_features(img_gray, mask):
    roi_img = img_gray.copy()
    roi_img[mask == 0] = 0

    bins = 32
    roi_img_binned = (roi_img / 256 * bins).astype(np.uint8)
    glcm = graycomatrix(
        roi_img_binned,
        distances=[1, 3],
        angles=[0, np.pi / 4, np.pi / 2, 3 * np.pi / 4],
        levels=bins,
        symmetric=True,
        normed=True
    )
    glcm[0, :, :, :] = 0
    glcm[:, 0, :, :] = 0
    sum_glcm = np.sum(glcm, axis=(0, 1), keepdims=True)
    with np.errstate(divide='ignore', invalid='ignore'):
        glcm = np.true_divide(glcm, sum_glcm)
        glcm[~np.isfinite(glcm)] = 0
        properties = ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation']

    glcm_features = []

    for prop in properties:
        feature_matrix = graycoprops(glcm, prop)

        mean_across_angles = np.mean(feature_matrix, axis=1)  # Risultato: array di 2 valori (Distanza 1, Distanza 3)
        glcm_features.extend(mean_across_angles)

        var_across_angles = np.var(feature_matrix, axis=1)
        glcm_features.extend(var_across_angles)

    return np.array(glcm_features)


# Returns a body-artifact clean and segmented lesion, with related masks
def cleaning_pipeline(unet, img_path, mask_path=None):
    img_bgr = cv2.imread(img_path)
    img_bgr = cv2.resize(img_bgr, (256, 256))

    img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    clean_bgr, _ = body_hair_removal(img_hsv, img_bgr)
    img_rgb = cv2.cvtColor(clean_bgr, cv2.COLOR_BGR2RGB)

    if mask_path is None:
        tensor = transform_to_tensor(img_rgb)
        tensor = tensor.unsqueeze(0)
        tensor = tensor.to(device)

        with torch.no_grad():
            logits = unet(tensor)
            probs = torch.sigmoid(logits)
            mask = (probs > 0.5).squeeze().cpu().numpy().astype(np.uint8)
            mask = mask * 255
    else:
        mask_loaded = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        mask_resized = cv2.resize(mask_loaded, (256, 256))
        _, mask = cv2.threshold(mask_resized, 127, 255, cv2.THRESH_BINARY)

    segmented_rgb = cv2.bitwise_and(img_rgb, img_rgb, mask=mask)

    aligned_img_rgb, aligned_mask = get_transrotated_img(segmented_rgb, mask)
    aligned_img_bgr = cv2.cvtColor(aligned_img_rgb, cv2.COLOR_RGB2BGR)
    return aligned_img_bgr, aligned_mask


# Returns the features necessary for assessing the complex structures presence
def extract_structure_features(img_bgr, mask, kmeans_model):
    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    haar_patches = extract_patches(img_gray, mask)
    haar_hist = build_haar_histogram(haar_patches, kmeans_model)
    lbp_features = extract_lbp_features(img_gray, mask)
    color_features = extract_color_features(img_bgr, mask)
    glcm_feaetures = extract_glcm_features(img_gray, mask)

    F_i = np.concatenate((haar_hist, lbp_features, color_features, glcm_feaetures))
    return F_i


# Returns the complete features extracted from the given aligned img and its mask
def feature_extraction(aligned_img_bgr, aligned_mask, kmeans_model, ra_svm, streaks_svm, pnet_svm, bluewv_svm):
    criteria = ['Streaks', 'Regression Areas', 'Blue-Whitish Veil', 'Pigment Network']

    features = {}
    features['Asymmetry'] = shape_asymmetry_index(aligned_mask)
    features['Color Asymmetry'] = color_asymmetry_index(aligned_img_bgr, aligned_mask)
    features['Border Irregularity'] = border_irregularity_index(aligned_mask, (
        aligned_mask.shape[1] // 2, aligned_mask.shape[0] // 2))

    iF = extract_structure_features(aligned_img_bgr, aligned_mask, kmeans_model)
    iF = iF.reshape(1, -1)

    regression_areas_b = ra_svm.predict(iF)[0]
    streaks_b = streaks_svm.predict(iF)[0]
    pnet_b = pnet_svm.predict(iF)[0]
    bluewv_b = bluewv_svm.predict(iF)[0]

    features["Regression Areas"] = regression_areas_b
    features["Streaks"] = streaks_b
    features["Blue-Whitish Veil"] = pnet_b
    features["Pigment Network"] = bluewv_b

    return features
