import argparse
from fiiter import median_filter, mean_filter, sharpen

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Applying image processing operation on a image')
    parser.add_argument("--image_path", help = "input path to image" )
    parser.add_argument("--option", type = int, help = " 1) median filter, 2) mean filter")
    args = parser.parse_args()

    if args.option == 1:
        median_filter(args.image_path)
    elif args.option == 2:
        mean_filter(args.image_path)
    elif args.option == 3:
        sharpen(args.image_path)