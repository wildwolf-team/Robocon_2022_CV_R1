#include <opencv2/opencv.hpp>
#include <iostream>

#include "Camera/mv_video_capture.hpp"
#include "TensorRTx/yolov5.hpp"
#include "angle_solve/basic_pnp.hpp"
#include "serial/uart_serial.hpp"
#include <fmt/color.h>
#include <fmt/core.h>

int main()
{
    mindvision::CameraParam CameraParams = mindvision::CameraParam(0,
                                                                 mindvision::RESOLUTION_1280_X_800,
                                                                 mindvision::EXPOSURE_10000);
    CameraParams.camera_exposuretime = 14000;
    mindvision::VideoCapture *mv_capture_ = new mindvision::VideoCapture(CameraParams);
    
    std::string engine_path = "/home/sweetdeath/Code/Robocon_2022_CV/models.engine";
    start(engine_path);

    cv::Mat src_img_;

    auto idntifier = fmt::format(fg(fmt::color::green) | fmt::emphasis::bold, "wolfvision");
    fmt::print("[{}] WolfVision config file path: {}\n", idntifier, "/home/sweetdeath/Code/Robocon_2022_CV/configs");
    basic_pnp::PnP pnp = basic_pnp::PnP(
        fmt::format("{}{}", "/home/sweetdeath/Code/Robocon_2022_CV/configs", "/camera_273.xml"), fmt::format("{}{}", "/home/sweetdeath/Code/Robocon_2022_CV/configs", "/basic_pnp_config.xml"));

    uart::SerialPort serial = uart::SerialPort(
        fmt::format("{}{}", "/home/sweetdeath/Code/Robocon_2022_CV/configs", "/uart_serial_config.xml"));

    while (mv_capture_->isindustryimgInput())
    {
        serial.updateReceiveInformation();
        src_img_ = mv_capture_->image();
        auto res = yolov5_v5_Rtx_start(src_img_);

        //对矩形的长宽比进行筛选，提取出最佳矩形
        float max_conf = .0;
        int max_conf_res_id = -1;
        for (size_t i = 0; i < res.size(); i++)
        {
            if(0.8 > res[i].bbox[2]/res[i].bbox[3] && res[i].bbox[2]/res[i].bbox[3] > 1.2)
                continue;
            if (res[i].conf > max_conf)
            {
                max_conf = res[i].conf;
                max_conf_res_id = i;
            }
        }

        //画出最佳矩形
        if (max_conf_res_id != -1)
        {
            // serial.updateReceiveInformation();
            cv::RotatedRect rotate_res(cv::Point2f(res[max_conf_res_id].bbox[0],res[max_conf_res_id].bbox[1]),cv::Size2f(res[max_conf_res_id].bbox[2],res[max_conf_res_id].bbox[3]), 0);
            pnp.solvePnP(30, 0, rotate_res);
            std::cout << "yaw:" << pnp.returnYawAngle() << "  pitch:" << pnp.returnPitchAngle() << std::endl;
            serial.updataWriteData(pnp.returnYawAngle(), pnp.returnPitchAngle(), pnp.returnDepth(), 1, 0);

            cv::Rect rect = get_rect(src_img_,res[max_conf_res_id].bbox);
            cv::rectangle(src_img_, rect, cv::Scalar(0x27, 0xC1, 0x36), 2);
            cv::putText(src_img_, std::to_string(pnp.returnDepth()), cv::Point(rect.x, rect.y - 1), cv::FONT_HERSHEY_PLAIN, 1.2, cv::Scalar(0xFF, 0xFF, 0xFF), 2);
        }
        else
        {
            serial.updataWriteData(pnp.returnYawAngle(), pnp.returnPitchAngle(), pnp.returnDepth(), 0, 0);
        }

        if (!src_img_.empty())
            cv::imshow("img", src_img_);
        if (cv::waitKey(1) == 'q')
            break;
        mv_capture_->cameraReleasebuff();
    }

    return 0;
}