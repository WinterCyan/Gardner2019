import argparse

def rend():
    parser = argparse.ArgumentParser()
    parser.add_argument('--param_file', type=str, default='params.txt')
    parser.add_argument('--dst_dir', type=str, default='sg_hdr')

    args = parser.parse_args()

if __name__ == '__main__':
    rend()
