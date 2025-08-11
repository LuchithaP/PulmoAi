import os
import cv2


def load_image_paths(positive_folder, negative_folder):
    if not os.path.exists(positive_folder):
        print(f"Error: Positive folder does not exist: {positive_folder}")
        return [], []
    if not os.path.exists(negative_folder):
        print(f"Error: Negative folder does not exist: {negative_folder}")
        return [], []

    image_paths = []
    labels = []

    def load_from_folder(folder, label):
        try:
            files = os.listdir(folder)
            if not files:
                print(f"Warning: No files found in folder: {folder}")
                return
            for filename in files:
                if filename.lower().endswith((".png", ".jpg", ".jpeg")):
                    img_path = os.path.join(folder, filename)
                    if os.path.isfile(img_path):
                        img = cv2.imread(img_path)
                        if img is not None:
                            image_paths.append(img_path)
                            labels.append(label)
                        else:
                            print(f"Failed to load image: {img_path}")
                    else:
                        print(f"Invalid file: {img_path}")
                else:
                    print(f"Skipped non-image file: {filename}")
        except Exception as e:
            print(f"Error accessing folder {folder}: {e}")

    load_from_folder(positive_folder, 1)
    load_from_folder(negative_folder, 0)
    print(f"Loaded {len(image_paths)} images")
    if not image_paths:
        print(
            "Warning: No valid images were loaded. Check folder paths and image files."
        )
    return image_paths, labels


if __name__ == "__main__":
    positive_folder = "Enter the path to the positive folder with COVID images here"
    negative_folder = "Enter the path to the negative folder with non-COVID images here"
    image_paths, labels = load_image_paths(positive_folder, negative_folder)
