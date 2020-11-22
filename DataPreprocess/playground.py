from DataPreprocess.ProcessEXR import *
from DataPreprocess.WarpUtils import *


if __name__ == '__main__':
    jpg_files = [f for f in listdir(hdr_jpgs_dir) if isfile(join(hdr_jpgs_dir, f)) and f.endswith(".jpg")]
    file_index = random.randrange(0, len(jpg_files))
    print(jpg_files[file_index])
    img_data = plt.imread(hdr_jpgs_dir+jpg_files[file_index])
    print(img_data.shape)
    # img_crop = crop(img_data)
