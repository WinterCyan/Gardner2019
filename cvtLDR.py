import argparse

def cvt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--src_dir', type=str, default='sg_hdr')
    parser.add_argument('--dst_dir', type=str, default='sg_ldr')

    args = parser.parse_args()

if __name__ == '__main__':
    cvt()
