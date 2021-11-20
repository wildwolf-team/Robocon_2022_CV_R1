#include <opencv2/opencv.hpp>
#include <iostream>

#include "Camera/mv_video_capture.hpp"
#include "TensorRTx/yolov5.hpp"

int main()
{
    mindvision::CameraParam CameraParams = mindvision::CameraParam(0,
                                                                 mindvision::RESOLUTION_1280_X_800,
                                                                 mindvision::EXPOSURE_10000);
    mindvision::VideoCapture *mv_capture_ = new mindvision::VideoCapture(CameraParams);
    
    std::string engine_path = "/home/sweetdeath/Code/Robocon_2022_CV/models.engine";
    start(engine_path);

    cv::Mat src_img_;

    while (mv_capture_->isindustryimgInput())
    {
        src_img_ = mv_capture_->image();
        auto res = yolov5_v5_Rtx_start(src_img_);

        // 绘制 矩形(rectangle) 和 类编号(class_id)
        for (size_t j = 0; j < res.size(); j++)
        { // res.size() 该图检测到多少个class
            cv::Rect r = get_rect(src_img_, res[j].bbox);
            cv::rectangle(src_img_, r, cv::Scalar(0x27, 0xC1, 0x36), 2);
            cv::putText(src_img_, std::to_string((int)res[j].class_id) + ":" + std::to_string(res[j].conf), cv::Point(r.x, r.y - 1), cv::FONT_HERSHEY_PLAIN, 1.2, cv::Scalar(0xFF, 0xFF, 0xFF), 2);
        }

        if (!src_img_.empty())
            cv::imshow("dafule", src_img_);
        if (cv::waitKey(1) == 'q')
            break;
        mv_capture_->cameraReleasebuff();
    }

    return 0;
}