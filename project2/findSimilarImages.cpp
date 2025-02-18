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

using namespace std;

/*
    Given the target filename, the directory of deep network embeddings csv, and the number of similar images to return, this function
    will retrieve vector value from csv file for the target image and all other images. It will then find the top N similar images to the target image using cosine similarity.
*/

void handleDeepNetworkEmbeddings(const string &target_filename, const string &directory, int num_similar_images)
{
    vector<char *> image_filenames;
    vector<vector<float>> image_data;
    vector<float> target_data;
    char buffer[256];
    FILE *fp;
    filesystem::path feature_file_path = filesystem::path(directory) / "ResNet18_olym.csv";
    string feature_file = feature_file_path.string();

    // Check if the pre-computed csv is existing
    fp = fopen(feature_file.c_str(), "r");
    if (!fp)
    {
        printf("Feature file not found. Please check the existence of file.\n");
        return;
    }

    // Read all image features from existing CSV file using read_image_data_csv
    read_image_data_csv(const_cast<char *>(feature_file.c_str()), image_filenames, image_data, 0);
    fclose(fp);

    // Read the target image feature
    for (int i = 0; i < image_filenames.size(); i++)
    {
        if (image_filenames[i] == target_filename)
        {
            target_data = image_data[i];
            break;
        }
    }

    // Find the top N similar images except the target image
    vector<pair<string, float>> distances;

    for (int i = 0; i < image_filenames.size(); i++)
    {
        if (image_filenames[i] != target_filename)
        {
            float similarity = matchDistance(target_data, image_data[i], cosineSimilarity);
            distances.push_back(make_pair(image_filenames[i], similarity));
        }
    }

    if (!distances.empty())
    {
        sort(distances.begin(), distances.end(),
             [](const pair<string, float> &a, const pair<string, float> &b)
             { return a.second < b.second; });

        for (size_t i = 0; i < min((size_t)num_similar_images, distances.size()); i++)
        {
            cout << distances[i].first << endl;
            // show the image
            {
                Mat img = imread(directory + "/" + distances[i].first);
                if (img.empty())
                {
                    cerr << "Error: Unable to read image file " << distances[i].first << endl;
                    return;
                }
                imshow("Image", img);
                waitKey(0);
            }
        }
    }
    else
    {
        cerr << "Error: No similar images found!" << endl;
    }
}

/*
  if the feature is not pre-computed yet, it will call processImages first. Otherwise, it will
  read the feature from the CSV file, and find the top N similar images to the target image.
  It takes in target filename for T, a directory of images as the database B, the feature function,
  the matching function, and the number of images N to return.
 */
int findSimilarImages(const string &csv_filename, const string &target_filename, const string &directory,
                      FeatureExtractor feature_method, MatchingMethod matching_method, int N)
{
    vector<string> image_filenames;
    vector<vector<float>> image_data;
    int echo_file = 0;
    vector<char *> filenames;
    vector<float> target_data;
    string full_path;
    string img_filename;
    char buffer[256];
    FILE *fp;

    if (feature_method = extractBanana)
    {
        handleDeepNetworkEmbeddings(target_filename, directory, N);
        return 0;
    }

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
        image_filenames.push_back(string(fname)); // Convert char* to string
    }
    fclose(fp);

    // Read the target image feature
    full_path = directory + "/" + target_filename;
    extractFeature(full_path, target_data, feature_method);

    // Find the top N similar images except the target image
    vector<pair<string, float>> distances;

    cout << "Image filenames count: " << image_filenames.size() << endl;
    cout << "Image data count: " << image_data.size() << endl;

    for (int i = 0; i < image_filenames.size(); i++)
    {
        if (image_filenames[i] != target_filename)
        {
            float similarity = matchDistance(target_data, image_data[i], matching_method);
            distances.push_back(make_pair(image_filenames[i], similarity));
        }
    }

    cout << "Distance count: " << distances.size() << endl;

    // Find smallest N distances
    if (!distances.empty())
    {
        sort(distances.begin(), distances.end(),
             [](const pair<string, float> &a, const pair<string, float> &b)
             { return a.second < b.second; });

        for (size_t i = 0; i < min((size_t)N, distances.size()); i++)
        {
            cout << distances[i].first << endl;
            // show the image
            {
                Mat img = imread(directory + "/" + distances[i].first);
                if (img.empty())
                {
                    cerr << "Error: Unable to read image file " << distances[i].first << endl;
                    return 0;
                }
                imshow("Image", img);
                waitKey(0);
            }
        }
    }
    else
    {
        cerr << "Error: No similar images found!" << endl;
    }

    return 0;
}

// Main function to run the program
int main(int argc, char *argv[])
{
    if (argc < 3)
    {
        cerr << "Usage: " << argv[0] << " <directory> <target_filename> <num_similar_images> <feature_choice> <matching_choice>" << endl;
        return 1;
    }
    string directory = argv[1];
    string target_filename = argv[2];
    int num_similar_images;
    int feature_choice;

    // Step 1: Input number of similar images to return
    cout << "Enter the number of similar images to return: ";
    cin >> num_similar_images;

    // Step 2: Choose feature extractor and distance metric
    cout << "Choose feature method:\n";
    cout << "1. 7x7 square in the middle of the image + sum of squared differences distance\n";
    cout << "2. Single color histogram of Hue (H) and Saturation (S) in HSV + histogram intersection distance\n";
    cout << "3. Two histogram of Hue (H) and Saturation (S) in HSV + weighted histogram intersection distance\n";
    cout << "4. Color histogram and texture histogram + weighted histogram intersection distance\n";
    cout << "5. Deep Network Embeddings + cosine similarity distance\n";
    cout << "6. Custom design: find outdoor scene + histogram intersection distance\n";
    cout << "7. Extension design: find banana + cosine similarity distance\n";
    cin >> feature_choice;

    // Define featureMethod based on user choice
    FeatureExtractor featureMethod;
    MatchingMethod matchingMethod;
    string method_name;
    if (feature_choice == 1)
    {
        featureMethod = extractCentralSquareFeature;
        matchingMethod = sum_of_squared_differences;
        method_name = "feature_method1";
    }
    else if (feature_choice == 2)
    {
        featureMethod = extractHSFeature;
        matchingMethod = histogramIntersection;
        method_name = "feature_method2";
    }
    else if (feature_choice == 3)
    {
        featureMethod = extractTwoHSVHistFeature;
        matchingMethod = twoHistogram;
        method_name = "feature_method3";
    }
    else if (feature_choice == 4)
    {
        featureMethod = extractColorTextureFeature;
        matchingMethod = oneColorOneTexture;
        method_name = "feature_method4";
    }
    else if (feature_choice == 5)
    {
        handleDeepNetworkEmbeddings(target_filename, directory, num_similar_images);
        return 0;
    }
    else if (feature_choice == 6)
    {
        featureMethod = extractOutdoors;
        matchingMethod = histogramIntersection;
        method_name = "feature_method6";
    }
    else if (feature_choice == 7)
    {
        featureMethod = extractBanana;
        matchingMethod = cosineSimilarity;
        method_name = "feature_method7";
    }
    else
    {
        cerr << "Error: Invalid feature choice." << endl;
        return 1;
    }

    filesystem::path csv_path = filesystem::path(directory) / (method_name + ".csv");
    string csv_filename = csv_path.string();

    // Find the top N similar images
    findSimilarImages(csv_filename, target_filename, directory, featureMethod, matchingMethod, num_similar_images);

    return 0;
}
