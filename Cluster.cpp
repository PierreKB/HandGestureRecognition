#include "Cluster.h"

int Cluster::m_ = 2;

Cluster::Cluster() {}

Cluster::Cluster(int id, std::vector<DataPoint>* data, std::vector<Cluster>* clusters)
	: id_(id), data_(data), clusters_(clusters)
{
	for (int i = 0; i < data_->size(); ++i)
	{
		unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
		std::default_random_engine g(seed);
		std::uniform_real_distribution<double> distribution(0.0, 1.0);

		membershipValues_.push_back(distribution(g));
	}

	UpdateCentroid();
}

Cluster::~Cluster() {}

void Cluster::UpdateCentroid()
{
	DataPoint numerator(std::vector<float>(data_->at(0).coord().size(), 0.0f));
	float denominator = 0.0f;


	for (int i = 0; i < data_->size(); ++i)
	{
		float membershipToM = std::pow(membershipValues_[i], m_);

		auto weighted = data_->at(i).scalarMultiplication(membershipToM);

		numerator = numerator.sum(weighted);
		denominator += membershipToM;
	}

	centroid_.coord(numerator.scalarMultiplication(1.0f / denominator).coord());
}

float Cluster::UpdateMembershipValues()
{
	float max = -1.0f;

	for (int i = 0; i < data_->size(); ++i)
	{
		float value = 0.0f;
		float numerator = data_->at(i).euclideanDistance(centroid_);

		for (int j = 0; j < clusters_->size(); ++j)
		{
			float denominator = data_->at(i).euclideanDistance(clusters_->at(j).centroid());
			value += std::pow(numerator / denominator, 2.0f / (float)(m_ - 1));

		}

		value = 1.0f / value;

		if (std::abs((float)(value - membershipValues_[i])) > max) { max = std::abs(value - membershipValues_[i]); }
		
		membershipValues_[i] = value;
	}

	return max;
}

float Cluster::ComputeMembershipValues(const DataPoint& point)
{
	float value = 0.0f;
	float numerator = point.euclideanDistance(centroid_);

	for (int j = 0; j < clusters_->size(); ++j)
	{
		float denominator = point.euclideanDistance(clusters_->at(j).centroid());
		value += std::pow(numerator / denominator, 2.0f / (float)(m_ - 1));

	}
	
	return (1.0f / value);
}

int Cluster::id() { return id_; }

DataPoint Cluster::centroid() { return centroid_; }

std::vector<float>* Cluster::membershipValues() { return &membershipValues_; }