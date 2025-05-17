#!/bin/bash
set -e

# "Flatten" a complicated folder structure by soft-linking the files within it under another single folder.
# E.g. you have a folder structure like this:
# path1/source_folder/
# |- folder1/
# |  |- img1.png
# |  |- ...
# |- folder2/sub-folder2/
# |  |- img2.png
# |  |- ...
# `- folder3/sub-folder3/sub-sub-folder3/
#    |- img3.png
#    |- ...
# Now you want to soft-link all those .png files into another folder `all-images/` for easy retrieving:
# path2/all-images/
# |- img1.png
# |- img2.png
# |- img3.png
# |- ...

# Get absolute paths for source and target directories
SOURCE_DIR=$(readlink -f "$SOURCE_DIR")
TARGET_DIR=$(readlink -f "$TARGET_DIR")

# Hardcoded source and target directories - MODIFY THESE PATHS AS NEEDED
SOURCE_DIR="path1/source_folder"
TARGET_DIR="path2/all-images"

# File pattern to search for - MODIFY THIS PATTERN AS NEEDED
FILE_PATTERN="*.nii.gz"

# Prepend source path prefix to the soft-link name to ensure uniqueness or not.
#   - 0: do not prepend, use file name only (RISK: name conflict of files from different directories)
#   - 1: prepend path prefix to the file name
ENSURE_UNIQUE=0

# Soft-link using relative path or not.
#   - 0: use real path
#   - 1: use relative path to the source position
RELATIVE_PATH=1

# Get absolute paths for source and target directories
SOURCE_DIR=$(readlink -f "$SOURCE_DIR")
TARGET_DIR=$(readlink -f "$TARGET_DIR")

# Check if source directory exists
if [ ! -d "$SOURCE_DIR" ]; then
    echo "Error: Source directory '$SOURCE_DIR' does not exist"
    exit 1
fi

# Create the target directory if it doesn't exist
mkdir -p "$TARGET_DIR"

# Navigate to the source directory
cd "$SOURCE_DIR"

# Find all files matching the pattern and create soft links with relative paths
find . -name "$FILE_PATTERN" | while read -r file; do
    # Extract just the filename
    filename=$(basename "$file")

    # Determine the target link name based on ENSURE_UNIQUE flag
    if [ $ENSURE_UNIQUE -eq 0 ]; then
        # Use the original filename (could cause overwriting if duplicates exist)
        target_name="${filename}"
    else
        # Create a unique name using directory structure as prefix
        dir_path=$(dirname "$file" | sed 's/[\/\.]/_/g' | sed 's/^_//')
        target_name="${dir_path}_${filename}"
    fi

    # Get the absolute path of the original file
    abs_file_path="$SOURCE_DIR/$file"

    # Calculate the relative path from target directory to the original file
    if [ $RELATIVE_PATH -eq 0 ]; then
        dest_path=$(realpath "$abs_file_path")
    else
        dest_path=$(realpath --relative-to="$TARGET_DIR" "$abs_file_path")
    fi

    # Create the symbolic link
    ln -sf "$dest_path" "$TARGET_DIR/$target_name"

    # echo "Created link: $TARGET_DIR/$target_name -> $rel_path"
done

# echo "All links created successfully!"
