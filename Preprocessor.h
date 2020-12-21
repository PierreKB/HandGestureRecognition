#pragma once
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
#include "Parameters.h"



class Preprocessor
{
public:
	Preprocessor(const Parameters& parameters);
	~Preprocessor();

	void Process(cv::Mat& image, cv::Mat& output, cv::Rect& rect);
	void ExtractFeatureVector(cv::Mat& image, cv::Rect& rect, std::vector<float>& features);

private:
	void Thresholding(cv::Mat& image, cv::Mat& output);
	void ExtractBlob(cv::Mat& image, cv::Mat& output, cv::Rect& rect);

	Parameters parameters_;
};