/*
  Given the target filename, the chosen feature, and the number of images to return, this function
  find top N similar images to the target image. If the feature is not pre-computed yet, it will
  call processImages first.
 */

#include <vector>
#include <string>
#include <cstdio>
#include <iostream>
#include <filesystem>
#include "csv_util.h"
#include "extractFeature.h"
#include "processImages.h"
#include "matchDistance.h"
#include <filesystem>

/*
  if the feature is not pre-computed yet, it will call processImages first. Otherwise, it will
  read the feature from the CSV file, and find the top N similar images to the target image.
  It takes in target filename for T, a directory of images as the database B, the feature function,
  the matching function, and the number of images N to return.
 */
int findSimilarImages(const std::string &csv_filename, const std::string &target_filename, const std::string &directory,
                      FeatureExtractor feature_method, MatchingMethod matching_method, int N)
{
    std::vector<std::string> image_filenames;
    std::vector<std::vector<float>> image_data;
    int echo_file = 0;
    std::vector<char *> filenames;
    std::vector<float> target_data;
    std::string full_path;
    std::string img_filename;
    char buffer[256];
    FILE *fp;

    // Check if the feature is pre-computed
    fp = fopen(csv_filename.c_str(), "r");
    if (!fp)
    {
        printf("Feature file not found. Computing features...\n");
        processImages(csv_filename, directory, feature_method);
        // Reopen the file after processing
        fp = fopen(csv_filename.c_str(), "r");
        if (!fp)
        {
            printf("Error: Failed to open newly generated CSV file.\n");
            return -1;
        }
    }
    char *csv_file_cstr = const_cast<char *>(csv_filename.c_str());

    // Read all image features from existing CSV file using read_image_data_csv
    read_image_data_csv(csv_file_cstr, filenames, image_data, echo_file);
    for (char *fname : filenames)
    {
        image_filenames.push_back(std::string(fname)); // Convert char* to std::string
    }
    fclose(fp);

    // Read the target image feature
    full_path = directory + "/" + target_filename;
    extractFeature(full_path, target_data, feature_method);

    // Find the top N similar images except the target image
    std::vector<std::pair<std::string, float>> similarity_scores;

    std::cout << "Image filenames count: " << image_filenames.size() << std::endl;
    std::cout << "Image data count: " << image_data.size() << std::endl;

    for (int i = 0; i < image_filenames.size(); i++)
    {
        if (image_filenames[i] != target_filename)
        {
            float similarity = matchDistance(target_data, image_data[i], matching_method);
            similarity_scores.push_back(std::make_pair(image_filenames[i], similarity));
        }
    }

    std::cout << "Similarity scores count: " << similarity_scores.size() << std::endl;

    // Only sort if similarity_scores is non-empty
    if (!similarity_scores.empty())
    {
        std::sort(similarity_scores.begin(), similarity_scores.end(),
                  [](const std::pair<std::string, float> &a, const std::pair<std::string, float> &b)
                  { return a.second > b.second; });

        // Print only the available elements
        for (size_t i = 0; i < std::min((size_t)N, similarity_scores.size()); i++)
        {
            std::cout << similarity_scores[i].first << std::endl;
        }
    }
    else
    {
        std::cerr << "Error: No similar images found!" << std::endl;
    }

    return 0;
}

// Main function to run the program
int main(int argc, char *argv[])
{
    if (argc < 3)
    {
        std::cerr << "Usage: " << argv[0] << " <directory> <target_filename> <num_similar_images> <feature_choice> <matching_choice>" << std::endl;
        return 1;
    }
    std::string directory = argv[1];
    std::string target_filename = argv[2];
    int num_similar_images;
    int feature_choice;
    int matching_choice;

    // Step 1: Input number of similar images to return
    std::cout << "Enter the number of similar images to return: ";
    std::cin >> num_similar_images;

    // Step 2: Choose feature method
    std::cout << "Choose feature method:\n";
    std::cout << "1. Method 1\n";
    std::cout << "2. Method 2\n";
    std::cin >> feature_choice;

    // Step 3: Choose matching method
    std::cout << "Choose matching method:\n";
    std::cout << "1. Matching Method 1\n";
    std::cout << "2. Matching Method 2\n";
    std::cin >> matching_choice;

    // Define featureMethod based on user choice
    FeatureExtractor featureMethod;
    std::string method_name;
    if (feature_choice == 1)
    {
        featureMethod = extractCentralSquareFeature;
        method_name = "feature_method1";
    }
    else if (feature_choice == 2)
    {
        featureMethod = extractColorHistogramFeature;
        method_name = "feature_method2";
    }

    std::filesystem::path csv_path = std::filesystem::path(directory) / (method_name + ".csv");
    std::string csv_filename = csv_path.string();

    // Define matchingMethod based on user choice
    MatchingMethod matchingMethod;
    if (matching_choice == 1)
    {
        matchingMethod = sum_of_squared_differences;
    }
    else if (matching_choice == 2)
    {
        matchingMethod = histogramIntersection;
    }

    // Find the top N similar images
    findSimilarImages(csv_filename, target_filename, directory, featureMethod, matchingMethod, num_similar_images);

    return 0;
}
