#include <iostream>
#include <fstream>
#include <vector>
#include <opencv2/ml.hpp>
#include <opencv2/opencv.hpp>

using namespace cv;

// Function to read IDX3-UBYTE files
std::vector<std::vector<unsigned char>> readIDX3UByteFile(const std::string& filename) {
    std::ifstream file(filename, std::ios::binary);

    if (!file) {
        std::cerr << "Failed to open the IDX3-UBYTE file." << std::endl;
        return {};
    }

    // Read the IDX3-UBYTE file header
    char magicNumber[4];
    char numImagesBytes[4];
    char numRowsBytes[4];
    char numColsBytes[4];

    file.read(magicNumber, 4);
    file.read(numImagesBytes, 4);
    file.read(numRowsBytes, 4);
    file.read(numColsBytes, 4);

    // Convert the header information from big-endian to native endianness
    int numImages = (static_cast<unsigned char>(numImagesBytes[0]) << 24) | 
                     (static_cast<unsigned char>(numImagesBytes[1]) << 16) | 
                     (static_cast<unsigned char>(numImagesBytes[2]) << 8) | 
                      static_cast<unsigned char>(numImagesBytes[3]);
    int numRows = (static_cast<unsigned char>(numRowsBytes[0]) << 24) | 
                   (static_cast<unsigned char>(numRowsBytes[1]) << 16) | 
                   (static_cast<unsigned char>(numRowsBytes[2]) << 8) | 
                    static_cast<unsigned char>(numRowsBytes[3]);
    int numCols = (static_cast<unsigned char>(numColsBytes[0]) << 24) | 
                   (static_cast<unsigned char>(numColsBytes[1]) << 16) | 
                   (static_cast<unsigned char>(numColsBytes[2]) << 8) | 
                    static_cast<unsigned char>(numColsBytes[3]);

    // Initialize a vector to store the images
    std::vector<std::vector<unsigned char>> images;

    for (int i = 0; i < numImages; i++) {
        // Read each image as a vector of bytes
        std::vector<unsigned char> image(numRows * numCols);
        file.read(reinterpret_cast<char*>(image.data()), numRows * numCols);

        images.push_back(image);
    }

    file.close();

    return images;
}

// Function to read IDX1-UBYTE label files
std::vector<int> readLabelFile(const std::string& filename) {
    std::ifstream file(filename, std::ios::binary);

    if (!file) {
        std::cerr << "Failed to open the IDX1-UBYTE file." << std::endl;
        return {};
    }

    // Read the IDX1-UBYTE file header
    char magicNumber[4];
    char numLabelsBytes[4];

    file.read(magicNumber, 4);
    file.read(numLabelsBytes, 4);

    // Convert the header information from big-endian to native endianness
    int numLabels = (static_cast<unsigned char>(numLabelsBytes[0]) << 24) | 
                     (static_cast<unsigned char>(numLabelsBytes[1]) << 16) | 
                     (static_cast<unsigned char>(numLabelsBytes[2]) << 8) | 
                      static_cast<unsigned char>(numLabelsBytes[3]);

    // Initialize a vector to store the labels
    std::vector<int> labels;

    for (int i = 0; i < numLabels; i++) {
        // Read each label as a single byte
        char labelByte;
        file.read(&labelByte, 1);

        labels.push_back(static_cast<int>(labelByte));
    }

    file.close();

    return labels;
}

int main() {
    // Paths to your dataset files
    std::string filename = "/Users/leeowen/Downloads/train-images.idx3-ubyte";
    std::string label_filename = "/Users/leeowen/Downloads/train-labels.idx1-ubyte";

    // Read image and label data
    std::vector<std::vector<unsigned char>> imagesFile = readIDX3UByteFile(filename);
    std::vector<int> labelsFile = readLabelFile(label_filename);

    // Check if data reading was successful
    if (imagesFile.empty() || labelsFile.empty()) {
        std::cerr << "Error: Failed to read dataset files." << std::endl;
        return -1;
    }

    // Get the number of samples and size of each input layer (assuming flattened image)
    int numSamples = imagesFile.size();
    int inputLayerSize = imagesFile[0].size();

    // Feature scaling
    cv::Mat trainingData(numSamples, inputLayerSize, CV_32F);
    cv::Mat labelData(numSamples, 1, CV_32S);

    // Scale and prepare training data and labels
    for (int i = 0; i < numSamples; i++) {
        // Create a temporary OpenCV matrix for each image
        cv::Mat image(1, inputLayerSize, CV_32F);

        // Scale each pixel value to the range [0, 1] brightness
        for (int j = 0; j < inputLayerSize; j++) {
            image.at<float>(0, j) = static_cast<float>(imagesFile[i][j]) / 255.0;  // Feature scaling (0 to 1)
        }

        // Copy the scaled image to the training data matrix
        image.copyTo(trainingData.row(i));

        // Set the label for the corresponding image
        labelData.at<int>(i, 0) = labelsFile[i];
    }

    // Create and configure SVM
    cv::Ptr<cv::ml::SVM> svm = cv::ml::SVM::create();
    svm->setType(cv::ml::SVM::C_SVC);

    // Use a Polynomial kernel with degree 4
    svm->setKernel(cv::ml::SVM::POLY);
    svm->setDegree(4);

    // Adjustment of the regularization parameter C and the coefficient for the kernel function
    svm->setC(1);
    svm->setCoef0(1);

    // Train SVM
    svm->train(trainingData, cv::ml::ROW_SAMPLE, labelData);

    // Save the trained SVM model
    svm->save("/Users/leeowen/Desktop/svm_model.xml");

    return 0;
}
