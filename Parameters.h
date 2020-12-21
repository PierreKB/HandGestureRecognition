#pragma once


struct Parameters
{
    Parameters() {}

    Parameters(int hueMin, int hueMax, int saturationMin, int saturationMax, int medianFilterSize, int morphKernelSize)
    {
        hueMin_ = hueMin;
        hueMax_ = hueMax;
        saturationMin_ = saturationMin;
        saturationMax_ = saturationMax;
        medianFilterSize_ = medianFilterSize;
        morphKernelSize_ = morphKernelSize;

    }

    int hueMin_;
    int hueMax_;
    int saturationMin_;
    int saturationMax_;
    int medianFilterSize_;
    int morphKernelSize_;
};