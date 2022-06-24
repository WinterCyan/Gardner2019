import argparse

def inference():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', type=str, default='model')
    parser.add_argument('--model', type=str, default='trained.pth')
    parser.add_argument('--src', type=str, default='test_scenes')
    parser.add_argument('--dst', type=str, default='params.txt')
    parser.add_argument('--timer', action='store_true', default=False)

    args = parser.parse_args()

    print(args.model_dir)
    print(args.model)
    print(args.src)
    print(args.dst)


if __name__ == '__main__':
    inference()
