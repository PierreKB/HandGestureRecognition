#include "Preprocessor.h"


Preprocessor::Preprocessor(const Parameters& parameters)
	: parameters_(parameters)
{}

Preprocessor::~Preprocessor()
{}

void Preprocessor::Process(cv::Mat& image, cv::Mat& output, cv::Rect& rect)
{
    cv::Mat thresholded;

    Thresholding(image, thresholded);
    ExtractBlob(thresholded, output, rect);
}

void Preprocessor::ExtractFeatureVector(cv::Mat& image, cv::Rect& rect, std::vector<float>& features)
{
    if (rect.width > rect.height) { features.push_back((float)rect.width / (float)rect.height); }
    else { features.push_back((float)rect.height / (float)rect.width); }

    int blockWidth = image.cols / 4;
    int blockHeigth = image.rows / 3;

    for (int y = 0; y < image.rows; y += blockHeigth)
    {
        for (int x = 0; x < image.cols; x += blockWidth)
        {
            cv::Mat block = image(cv::Range(y, std::min(y + blockHeigth, image.rows)), cv::Range(x, std::min(x + blockWidth, image.cols)));
            
            cv::Scalar mean = cv::mean(block);

            features.push_back(mean[0]);
        }
    }
}



void Preprocessor::Thresholding(cv::Mat& image, cv::Mat& output)
{
    cv::Mat hsv;

    cv::cvtColor(image, hsv, cv::COLOR_BGR2HSV);
    output = cv::Mat::zeros(hsv.size(), CV_8UC1);

    //Segment skin region
    for (int y = 0; y < hsv.rows; ++y)
    {
        cv::Vec3b* pixel = hsv.ptr<cv::Vec3b>(y);

        for (int x = 0; x < hsv.cols; ++x)
        {
            if (pixel[x][0] >= parameters_.hueMin_ && pixel[x][0] <= parameters_.hueMax_ && pixel[x][1] >= parameters_.saturationMin_ && pixel[x][1] <= parameters_.saturationMax_)
            {
                output.at<uchar>(y, x) = 255;
            }
        }
    }

    //Noise removal
    medianBlur(output, output, 5);

    cv::Mat elt = cv::getStructuringElement(1, cv::Size(parameters_.morphKernelSize_, parameters_.morphKernelSize_));
    morphologyEx(output, output, 2, elt);
    morphologyEx(output, output, 3, elt);
}

void Preprocessor::ExtractBlob(cv::Mat& image, cv::Mat& output, cv::Rect& boundingRect)
{
    std::vector<std::vector<cv::Point2d>> blobs;
    std::vector<cv::Rect> rects;

    cv::Mat labeledImage;
    image.convertTo(labeledImage, CV_32FC1, 1.0 / 255.0);

    int label = 2;

    int biggest = 0;
    int previousSize = -1;

    for (int y = 0; y < image.rows; y++) 
    {
        for (int x = 0; x < image.cols; x++)
        {
            if ((int)labeledImage.at<float>(y, x) != 1) { continue; }

            cv::Rect rect;
            cv::floodFill(labeledImage, cv::Point(x, y), cv::Scalar(label), &rect, cv::Scalar(0), cv::Scalar(0), 8);

            std::vector<cv::Point2d> blob;

            for (int i = rect.y; i < (rect.y + rect.height); i++) 
            {
                for (int j = rect.x; j < (rect.x + rect.width); j++) 
                {
                    if ((int)labeledImage.at<float>(i, j) != label) { continue; }

                    blob.push_back(cv::Point(j, i));
                }
            }

            blobs.push_back(blob);
            rects.push_back(rect);

            label++;

            if ((int)blob.size() > previousSize)
            {
                biggest = label - 3;
                previousSize = blob.size();
            }
        }
    }

    output = cv::Mat::zeros(image.size(), CV_8UC1);

    for (auto& point : blobs[biggest])
    {
        output.at<uchar>(point) = 255;
    }

    boundingRect = rects[biggest];
    rectangle(output, boundingRect, cv::Scalar(255));
}