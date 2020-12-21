#pragma once
#include <vector>
#include <algorithm>
#include <random>
#include <cmath>
#include <chrono>
#include "DataPoint.h"

class Cluster
{
public:
	Cluster();
	Cluster(int id, std::vector<DataPoint>* data, std::vector<Cluster>* clusters);

	~Cluster();

	void UpdateCentroid();
	float UpdateMembershipValues();

	//Compute the membership value for a particular point
	float ComputeMembershipValues(const DataPoint& point);

	int id();

	DataPoint centroid();
	std::vector<float>* membershipValues();
	
	static int m_;

private:
	int id_;

	DataPoint centroid_;
	std::vector<float> membershipValues_;
	std::vector<DataPoint>* data_;
	std::vector<Cluster>* clusters_;
};