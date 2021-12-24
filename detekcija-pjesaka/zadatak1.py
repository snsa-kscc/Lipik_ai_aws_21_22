import os
import cv2
from skimage.feature import hog
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier

dataset_path = 'dataset'

mean_image_width = 101
mean_image_height = 264


def load_ped_dataset(images_folder):
    ped_image_dataset = os.path.join(dataset_path, images_folder)

    ped_hog_list = []

    for file in os.listdir(ped_image_dataset):
        if file.endswith('.png'):
            image_name = os.path.join(ped_image_dataset, file)
            image_ped = cv2.imread(image_name, cv2.IMREAD_GRAYSCALE)
            image_ped = cv2.resize(
                image_ped, (mean_image_width, mean_image_height))

            HOG_desc, hog_image = hog(image_ped, visualize=True)

            # fig, axes = plt.subplots(1, 2)

            # axes[0].imshow(image_ped, cmap='gray')
            # axes[1].imshow(hog_image, cmap='gray')
            # plt.show()

            ped_hog_list.append(HOG_desc)

    ped_hog_list = np.array(ped_hog_list)

    return ped_hog_list


ped_dataset = load_ped_dataset('ped')
no_ped_dataset = load_ped_dataset('no_ped')

pedestrian_labels = np.ones((ped_dataset.shape[0]))
no_pedestrian_labels = np.zeros(no_ped_dataset.shape[0])

x = np.concatenate((ped_dataset, no_ped_dataset))
y = np.concatenate((pedestrian_labels, no_pedestrian_labels))

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

classifier = MLPClassifier(random_state=1)
classifier.fit(x_train, y_train)
score = classifier.score(x_test, y_test)

print(score)

test_img = cv2.imread("dataset/test_img.png", cv2.IMREAD_GRAYSCALE)
img_height = test_img.shape[0]
img_width = test_img.shape[1]

max_pedestrian_probability = 0
final_top_left_bb = (0, 0)
final_bottom_right_bb = (0, 0)

for i in range(0, img_height - mean_image_height - 1, 10):
    for j in range(0, img_width - mean_image_width - 1, 10):
        roi = test_img[i:i+mean_image_height, j:j+mean_image_width]
        hog_desc, hog_image = hog(roi, visualize=True)

        roi_probabilities = classifier.predict_proba(hog_desc.reshape(1, -1))
        pedestrian_probabilities = roi_probabilities[0][1]

        if pedestrian_probabilities > max_pedestrian_probability:
            max_pedestrian_probability = pedestrian_probabilities
            final_top_left_bb = (j, i)
            final_bottom_right_bb = (i+mean_image_height, j+mean_image_width)


detection_img = cv2.rectangle(
    test_img, final_top_left_bb, final_bottom_right_bb, 255, 2)
cv2.imshow("detection", detection_img)
cv2.waitKey()
cv2.destroyAllWindows()

print("hi mom")
