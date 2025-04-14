#  last update: 2025.1.15 by fjl
import os
import shutil


def copy_images_to_new_folder(source_folder, destination_folder):
    """
    Copy all the pictures in all the subfolders under the parent folder to a new folder.

    Args:
        source_folder (str): Source folder path, including subfolders
        destination_folder (str): Target folder path, where all the pictures are stored
    """
    os.makedirs(destination_folder, exist_ok=True)

    supported_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tif'}

    # for all pictures in all the subfolders
    for root, _, files in os.walk(source_folder):
        for file in files:
            # check is picture?
            if os.path.splitext(file)[1].lower() in supported_extensions:
                source_path = os.path.join(root, file)
                destination_path = os.path.join(destination_folder, file)

                # If there is a file with the same name in the target path, then add a numeric suffix.
                base_name, extension = os.path.splitext(file)
                counter = 1
                while os.path.exists(destination_path):
                    destination_path = os.path.join(
                        destination_folder, f"{base_name}_{counter}{extension}"
                    )
                    counter += 1

                # 复制文件
                shutil.copy2(source_path, destination_path)
                print(f"Copy: {source_path} -> {destination_path}")

    print(f"All pictures have been copied to {destination_folder}。")


# 示例用法
source_folder = '~'  # Replace with the actual path of the source folder.
destination_folder = '~'  # Replace with the actual path of the target folder.
copy_images_to_new_folder(source_folder, destination_folder)