#include "Classifier.h"


Classifier::Classifier(std::vector<Cluster>* clusters, const std::map<int, std::string>& clustersIndexNameMap)
	: clusters_(clusters), clustersIndexNameMap_(clustersIndexNameMap)
{}

Classifier::~Classifier() {}

std::map<std::string, float> Classifier::Classify(const std::vector<DataPoint>& testData, const std::vector<std::string>& testLabels,
	std::vector<std::string>& foundLabels, std::vector<float>& foundMembership, float threshold)
{
	int clusterNumbers = clusters_->size();
	auto clusters = *clusters_;

	for (int i = 0; i < testData.size(); ++i)
	{
		std::vector<float> memberships;

		for (int j = 0; j < clusterNumbers; ++j)
		{
			memberships.push_back(clusters[j].ComputeMembershipValues(testData[i]));
		}

		auto max = std::max_element(memberships.begin(), memberships.end());

		for (int k = 0; k < memberships.size(); ++k)
		{
			if (memberships[k] == (float)(*max))
			{
				if (memberships[k] >= threshold) { foundLabels.push_back(clustersIndexNameMap_[k]); }
				else { foundLabels.push_back("Unclassified"); }

				foundMembership.push_back(memberships[k]);

			}
		}
	}

	std::map<std::string, float> recognitionRate;

	for (auto it : clustersIndexNameMap_)
	{
		recognitionRate[it.second] = 0.0f;
	}

	for (int i = 0; i < foundLabels.size(); ++i)
	{
		for (auto& it : recognitionRate)
		{
			if (testLabels[i] == it.first)
			{
				if (foundLabels[i] == testLabels[i])
				{
					recognitionRate[it.first] += 1;
				}
			}
		}
	}

	return recognitionRate;
}

void Classifier::Classify(const std::vector<DataPoint>& testData, std::vector<std::string>& foundLabels, std::vector<float>& foundMembership, float threshold)
{
	int clusterNumbers = clusters_->size();
	auto clusters = *clusters_;

	for (int i = 0; i < testData.size(); ++i)
	{
		std::vector<float> memberships;

		for (int j = 0; j < clusterNumbers; ++j)
		{
			memberships.push_back(clusters[j].ComputeMembershipValues(testData[i]));
		}

		auto max = std::max_element(memberships.begin(), memberships.end());

		for (int k = 0; k < memberships.size(); ++k)
		{
			if (memberships[k] == (float)(*max))
			{
				if (memberships[k] >= threshold) { foundLabels.push_back(clustersIndexNameMap_[k]); }
				else { foundLabels.push_back("Unclassified"); }

				foundMembership.push_back(memberships[k]);

			}
		}
	}
}