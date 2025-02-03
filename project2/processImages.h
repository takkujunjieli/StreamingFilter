#ifndef PROCESS_IMAGES_H
#define PROCESS_IMAGES_H

#include <string>
#include <vector>

std::vector<std::string> getImagesInDirectory(const std::string &directory);
int processImages(const std::string &filename, const std::string &directory, FeatureExtractor featureMethod);

#endif