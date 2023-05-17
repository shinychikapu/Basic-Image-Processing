import argparse
from fiiter import median_filter, mean_filter, sharpen, otsu_thresholding, edge_detection, gaussian_smooth

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Applying image processing operation on a image')
    parser.add_argument("--image_path", help = "input path to image" )
    parser.add_argument("--option", type = int, help = " 1) Median filter, 2) Mean filter, 3) Gaussian Blur, 4) Sharpen, 5)Otsu's thresholding, 6) Edge DetectionS")
    args = parser.parse_args()

    if args.option == 1:
        median_filter(args.image_path)
    elif args.option == 2:
        mean_filter(args.image_path)
    elif args.option == 3:
        gaussian_smooth(args.image_path) 
    elif args.option == 4:
        sharpen(args.image_path)  
    elif args.option == 5:
        otsu_thresholding(args.image_path) 
    elif args.option == 6:
        edge_detection(args.image_path)