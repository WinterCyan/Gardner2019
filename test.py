import argparse

def test():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', type=str, default='model')
    parser.add_argument('--model', type=str, default='trained.pth')
    parser.add_argument('--test_datadir', type=str, default='eval_scenes')
    parser.add_argument('--target_file', type=str, default='params_target.txt')

    args = parser.parse_args()

    print(args.model_dir)
    print(args.model)
    print(args.src)
    print(args.dst)


if __name__ == '__main__':
    test()
