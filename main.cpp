#include <opencv2/opencv.hpp>
#include <iostream>

#include "mv_video_capture.hpp"

int main()
{
    mindvision::CameraParam CameraParams = mindvision::CameraParam(0,
                                                                 mindvision::RESOLUTION_1280_X_800,
                                                                 mindvision::EXPOSURE_10000);
    mindvision::VideoCapture *mv_capture_ = new mindvision::VideoCapture(CameraParams);
    
    cv::Mat src_img_;

    while (mv_capture_->isindustryimgInput())
    {
        src_img_ = mv_capture_->image();
        if (!src_img_.empty())
        cv::imshow("dafule", src_img_);
        if (cv::waitKey(1) == 'q')
            break;
        mv_capture_->cameraReleasebuff();
    }

    return 0;
}