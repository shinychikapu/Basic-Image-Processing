import argparse
from meanFilter import mean_filter

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Applying image processing operation on a image')
    parser.add_argument("--image_path", help = "input path to image" )
    
    args = parser.parse_args()

    mean_filter(args.image_path)

