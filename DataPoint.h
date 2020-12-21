#pragma once
#include <vector>
#include <iostream>

struct DataPoint
{
public:
	DataPoint() {}

	DataPoint(const std::vector<float>& coord)
		: coord_(coord)
	{}

	DataPoint sum(const DataPoint& datapoint)
	{
		if (size() != datapoint.size()) { std::cout << "Can't sum vectors of different dimensionality" << std::endl; return DataPoint(); }

		std::vector<float> output;
		auto coords = datapoint.coord();

		for (int i = 0; i < coord_.size(); ++i)
		{
			output.push_back(coord_[i] + coords[i]);
		}

		return DataPoint(output);
	}

	DataPoint scalarMultiplication(float a)
	{
		std::vector<float> output;

		for (int i = 0; i < coord_.size(); ++i)
		{
			output.push_back(coord_[i] * a);
		}

		return DataPoint(output);
	}

	float euclideanDistance(const DataPoint& datapoint) const
	{
		if (coord_.size() != datapoint.size()) 
		{ 
			std::cout << "Can't compute the distance between vectors of different dimensionality" << std::endl; 
			this->value();
			std::cout << std::endl;
			datapoint.value();

			return -1.0f;
		}

		float distance = 0.0f;
		auto coords = datapoint.coord();

		for (int i = 0; i < coord_.size(); ++i)
		{
			distance += std::pow(coord_[i] - coords[i], 2);
		}

		return sqrt(distance);
	}

	std::vector<float> coord() const { return coord_; }

	void coord(const std::vector<float>& coord) { coord_ = coord; }

	int size() const { return coord_.size(); }

	void value() const
	{
		std::cout << "[ ";

		for (auto& c : coord_)
		{
			std::cout << c << "  ";
		}

		std::cout << " ]";
	}

private:
	std::vector<float> coord_;
};