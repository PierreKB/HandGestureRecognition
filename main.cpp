#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
#include <iomanip>
#include <map>

#include "Cluster.h"
#include "Preprocessor.h"
#include "Classifier.h"





int main(int argc, char** argv)
{
	bool launchDemo = true;

    //Get training data
    std::vector<cv::String> trainingFilesNames;
	std::vector<cv::String> testFilesNames;
	std::vector<std::string> testLabels;

    std::string base= "./dataset/training/";

    std::string folders[3] = { "0000", "0001", "0002"};
    std::string subfolders[20] = { "0000", "0001", "0002", "0003", "0004", "0005", "0006", "0008", "0009", "0010", "0011", "0012", "0013", "0014", "0015", "0016", "0017", "0018", "0019" };
	std::map<std::string, std::string> folderLabelMap = { {"0000", "Left"}, {"0001", "Right"}, {"0002", "Compact"} };


    for (int set = 1; set <= 4; ++set)
    {
        for (auto& folder : folders)
        {
            for (auto& subfolder : subfolders)
            {
                std::string path = base + "set " + std::to_string(set) + "/" + folder + "/" + subfolder + "/*.jpg";
                
                std::vector<cv::String> filesNames;
                cv::glob(path, filesNames, true);

                trainingFilesNames.push_back(filesNames[filesNames.size() - 1]);
            }
        }
    }

	for (auto& folder : folders)
	{
		for (auto& subfolder : subfolders)
		{
			std::string path = "./dataset/test/" + folder + "/" + subfolder + "/*.jpg";

			std::vector<cv::String> filesNames;
			cv::glob(path, filesNames, true);

			testFilesNames.push_back(filesNames[filesNames.size() - 1]);
			testLabels.push_back(folderLabelMap[folder]);
		}
	}
    
	//Data preprocessing
	Parameters parameters(0, (0.14 * 179), (0.1 * 255), (0.7 * 255), 5, 5);
	Preprocessor preprocessor(parameters);

	std::vector<DataPoint> trainingData;

    for (int i = 0; i < trainingFilesNames.size(); ++i)
    {
        cv::Mat image = cv::imread(trainingFilesNames[i], cv::IMREAD_COLOR);

        if (image.empty())
        {
            std::cout << "Could not open or find the image" << std::endl;
            return -1;
        }

        cv::Mat output;
        cv::Rect rect;
        std::vector<float> features;

        preprocessor.Process(image, output, rect);
        preprocessor.ExtractFeatureVector(output, rect, features);

		trainingData.push_back(DataPoint(features));  
    }


	int clusterNumbers = 3;
	std::vector<Cluster> clusters;

	for (int i = 0; i < clusterNumbers; ++i) 
	{
		clusters.push_back(Cluster(i, &trainingData, &clusters));
	}

	Cluster::m_ = 2;

	//Clustering
	int iteration = 1;
	while(true)
	{
		std::vector<float> diff;

		for (int j = 0; j < clusterNumbers; ++j)
		{
			clusters[j].UpdateCentroid();
			diff.push_back(clusters[j].UpdateMembershipValues());
		}

		auto max = std::max_element(diff.begin(), diff.end());

		std::cout << "Variation max: " << std::setw(16) << *max << "      Iteration: " << std::setw(4) << iteration << std::endl;
		

		if ((float)(*max) < 0.001f) { break; }
		++iteration;
	}

	//Find which computed cluster correspond to which real world class
	std::map<std::string, std::vector<std::pair<int, int>>> locatedInDataSet;
	std::map<std::string, DataPoint> meanMap;

	locatedInDataSet = {
		{"Left", { {0, 19}, {60, 79}, {120, 139}, {180, 199} } },
		{"Right", { {20, 39}, {80, 99}, {140, 159}, {200, 219} } },
		{"Compact", { {40, 59}, {100, 119}, {160, 179}, {220, 239} } }
	};


	for (auto& it1 : locatedInDataSet)
	{
		meanMap[it1.first] = DataPoint(std::vector<float>(clusterNumbers, 0.0f));

		for (auto& it2 : it1.second)
		{
			for (int i = it2.first; i <= it2.second; ++i)
			{
				for (int j = 0; j < clusterNumbers; ++j)
				{
					std::vector<float> zeros(clusterNumbers, 0.0f);

					zeros[j] = clusters[j].membershipValues()->at(i);
					meanMap[it1.first] = meanMap[it1.first].sum(DataPoint(zeros));
				}
			}
		}

		meanMap[it1.first] = meanMap[it1.first].scalarMultiplication(1.0f / 80.0f);
	}

	//Get prototype features vectors for each class
	std::map<std::string, DataPoint> prototypeFeaturesVectors;
	std::map<int, std::string> clustersIndexNameMap;

	for (auto& it1 : meanMap)
	{
		auto vec = it1.second.coord();
		auto max = std::max_element(vec.begin(), vec.end());

		for (int cluster = 0; cluster < clusterNumbers; ++cluster)
		{
			if (vec[cluster] == (float)(*max))
			{
				prototypeFeaturesVectors[it1.first] = clusters[cluster].centroid();
				clustersIndexNameMap[cluster] = it1.first;
			}
		}
	}

	std::cout << "Prototype feature vectors: " << std::endl;

	for (auto& vector : prototypeFeaturesVectors)
	{
		std::cout << vector.first << ": ";
		vector.second.value();
		std::cout << std::endl;
	}


	//Now test the classifier
	//Load and preprocess test data
	//Data preprocessing
	std::vector<DataPoint> testData;
	std::vector<std::string> foundLabels;
	std::vector<float> foundMembership;


	for (int i = 0; i < testFilesNames.size(); ++i)
	{
		cv::Mat image = cv::imread(testFilesNames[i], cv::IMREAD_COLOR);

		if (image.empty())
		{
			std::cout << "Could not open or find the image" << std::endl;
			return -1;
		}

		cv::Mat output;
		cv::Rect rect;
		std::vector<float> features;

		preprocessor.Process(image, output, rect);
		preprocessor.ExtractFeatureVector(output, rect, features);

		testData.push_back(DataPoint(features));
	}


	Classifier classifier(&clusters, clustersIndexNameMap);

	auto recognitionRate = classifier.Classify(testData, testLabels, foundLabels, foundMembership, 0.4);
	
	for (int i = 0; i < foundLabels.size(); ++i)
	{
		std::cout << "True label: " << std::setw(8) << testLabels[i] 
			<< " | Found label: " << std::setw(14) << foundLabels[i] 
			<< "| Membership value: " << std::setw(5) << foundMembership[i] << std::endl;
	}

	std::cout << std::endl;

	for (auto& it : recognitionRate)
	{
		std::cout <<"Class: " << std::setw(8) << it.first << " | recognition rate: " << (it.second / 20.0f) * 100 << "%" << std::endl;
	}

	//Now try it real time
	if (launchDemo)
	{
		cv::VideoCapture camera(0);

		cv::Mat imageRect = cv::Mat::zeros(cv::Size(500, 500), CV_8UC1);
		cv::Mat frame;
		int elapsed = 0;
		cv::Point2d rectPosition(200, 200);

		cv::rectangle(imageRect, cv::Rect2d(200, 200, 50, 50), cv::Scalar(255), cv::FILLED);

		if (!camera.isOpened()) {
			std::cerr << "Unable to open the camera!" << std::endl;
			return 0;
		}

		cv::namedWindow("Moving Rectangle");
		cv::namedWindow("Capture");

		while (camera.read(frame))
		{
			++elapsed;

			if (elapsed >= 30)
			{
				elapsed = 0;

				std::vector<std::string> labels;
				std::vector<float> membership;

				cv::Mat output;
				cv::Rect rect;
				std::vector<float> features;

				preprocessor.Process(frame, output, rect);
				preprocessor.ExtractFeatureVector(output, rect, features);

				DataPoint point(features);

				Classifier classifier(&clusters, clustersIndexNameMap);
				classifier.Classify({ point }, labels, membership, 0.4);

				if (labels[0] != "Unclassified" && labels[0] != "Compact")
				{
					imageRect = cv::Mat::zeros(cv::Size(500, 500), CV_8UC1);

					if (labels[0] == "Left") { rectPosition -= cv::Point2d(30, 0); }
					else { rectPosition += cv::Point2d(30, 0); }

					cv::rectangle(imageRect, cv::Rect2d(rectPosition.x, rectPosition.y, 50, 50), cv::Scalar(255), cv::FILLED);
				}
			}

			cv::imshow("Moving rect", imageRect);
			cv::imshow("Capture", frame);

			int keyboard = cv::waitKey(30);
			if (keyboard == 'q' || keyboard == 27)
				break;
		}
	}

    
	//cv::waitKey(0);
    return 0;
}