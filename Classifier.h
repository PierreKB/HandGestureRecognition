#pragma once
#include <map>
#include "Cluster.h"


class Classifier
{
public:
	Classifier(std::vector<Cluster>* clusters, const std::map<int, std::string>& clustersIndexNameMap);
	~Classifier();

	std::map<std::string, float> Classify(const std::vector<DataPoint>& testData, const std::vector<std::string>& testLabels, 
		std::vector<std::string>& foundLabels, std::vector<float>& foundMembership, float threshold);
	
	void Classify(const std::vector<DataPoint>& testData, std::vector<std::string>& foundLabels, std::vector<float>& foundMembership, float threshold);

private:
	std::vector<Cluster>* clusters_;
	std::map<int, std::string> clustersIndexNameMap_;
};