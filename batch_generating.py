import os
import shutil

def generate_image_sets():
  
    base_dir = "resources"
    dataset_dir = os.path.join(base_dir, "dataset")

    base_filenames = ["scene_key.pgm", "scene_key_rot90.pgm", "scene_key_rot180.pgm", "scene_key_rot270.pgm"]

    base_images = [os.path.join(base_dir, f) for f in base_filenames]

    for img in base_images:
        if not os.path.exists(img):
            raise FileNotFoundError(f"Missing specificated image: {img}")

    print(f"Using the following images from '{base_dir}': {base_filenames}")

    os.makedirs(dataset_dir, exist_ok=True)

    targets = [64, 128, 512, 1024]

    for target in targets:
        set_dir = os.path.join(dataset_dir, f"set_{target}")
        os.makedirs(set_dir, exist_ok=True)

        count = 0
        while count < target:
            for img in base_images:
                if count >= target:
                    break
                filename = f"img_{count+1:04d}" + os.path.splitext(img)[1]
                dst_path = os.path.join(set_dir, filename)
                shutil.copy(img, dst_path)
                count += 1

        print(f"Created {set_dir} with {count} images.")

if __name__ == "__main__":
    generate_image_sets()
