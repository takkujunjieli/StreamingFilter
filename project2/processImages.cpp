#include <vector>
#include <string>
#include <cstdio>
#include <iostream>
#include <filesystem>
#include "csv_util.h"
#include "extractFeature.h"
#include "processImages.h"

/*
  Given a filename, a directory and chosen feature extraction method, this function
  write the feature of all images in that directory to the given CSV file.
*/

int processImages(const std::string &csv_filename, const std::string &directory, FeatureExtractor featureMethod)
{
    std::vector<std::string> image_filenames;
    std::vector<std::vector<float>> image_data;
    std::string full_path;
    char buffer[256];

    FILE *fp = fopen(csv_filename.c_str(), "a");
    // if the file does not exist, create a new file
    if (!fp)
    {
        fp = fopen(csv_filename.c_str(), "w"); // Create new file if it doesn't exist
        if (!fp)
        {
            std::cerr << "Error: Unable to open or create CSV file: " << csv_filename << std::endl;
            return -1;
        }
    }

    // Get the list of image files in the directory
    image_filenames = getImagesInDirectory(directory);

    // Process each image file
    for (const auto &img_filename : image_filenames)
    {
        std::vector<float> image_features;
        full_path = directory + "/" + img_filename;
        extractFeature(full_path, image_features, featureMethod);
        append_image_data_csv(const_cast<char *>(csv_filename.c_str()),
                              const_cast<char *>(img_filename.c_str()),
                              image_features, 0);
    }

    fclose(fp);

    return 0;
}

std::vector<std::string> getImagesInDirectory(const std::string &directory)
{
    std::vector<std::string> image_filenames;
    for (const auto &entry : std::filesystem::directory_iterator(directory))
    {
        if (entry.is_regular_file())
        {
            std::string extension = entry.path().extension().string();
            if (extension == ".jpg" || extension == ".png")
            {
                image_filenames.push_back(entry.path().filename().string());
            }
        }
    }
    return image_filenames;
}