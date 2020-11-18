from DataPreprocess.ProcessEXR import *
from DataPreprocess.WarpUtils import *


if __name__ == '__main__':
    jpg_files = [f for f in listdir(hdr_jpgs_dir) if isfile(join(hdr_jpgs_dir, f)) and f.endswith(".jpg")]
    file_index = random.randrange(0, len(jpg_files))
    print(jpg_files[file_index])
    img_data = plt.imread(hdr_jpgs_dir+jpg_files[file_index])
    print(img_data.shape)
    # img_crop = crop(img_data)
    img_crop, img_crop2 = crop_rec(img_data)

    max_v = np.amax(img_crop)
    min_v = np.amin(img_crop)
    corrected = (img_crop-min_v)/(max_v-min_v)

    max_v = np.amax(img_crop2)
    min_v = np.amin(img_crop2)
    corrected2 = (img_crop2-min_v)/(max_v-min_v)

    plt.imsave("../Files/cropped_img.jpg", corrected)
    plt.imsave("../Files/cropped_img2.jpg", corrected2)

