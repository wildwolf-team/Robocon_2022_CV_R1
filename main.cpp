#include <opencv2/opencv.hpp>
#include <iostream>
#include <fmt/color.h>
#include <fmt/core.h>

#include "Camera/mv_video_capture.hpp"
#include "TensorRTx/yolov5.hpp"
#include "angle_solve/basic_pnp.hpp"
#include "serial/uart_serial.hpp"

//对矩形的长宽比进行筛选，提取出最佳矩形
cv::Rect FilterRect(std::vector<Yolo::Detection> res, cv::Mat img)
{
    float max_conf = .0;
    int max_conf_res_id = -1;
    for (size_t i = 0; i < res.size(); i++)
    {
        if (0.8 > res[i].bbox[2] / res[i].bbox[3] && res[i].bbox[2] / res[i].bbox[3] > 1.2)
            continue;
        if (res[i].conf > max_conf)
        {
            max_conf = res[i].conf;
            max_conf_res_id = i;
        }
    }
    return get_rect(img, res[max_conf_res_id].bbox);
}

int main(int argc, char *argv[])
{
    mindvision::CameraParam CameraParams = mindvision::CameraParam(0,
                                                                   mindvision::RESOLUTION_1280_X_800,
                                                                   20000);
    mindvision::VideoCapture mv_capture_ = mindvision::VideoCapture(CameraParams);

    start(fmt::format("{}{}", SOURCE_PATH, "/Models/RCBall3.engine"));

    cv::Mat src_img_;

    basic_pnp::PnP pnp = basic_pnp::PnP(fmt::format("{}{}", CONFIG_FILE_PATH, "/camera_273.xml"),
                                        fmt::format("{}{}", CONFIG_FILE_PATH, "/basic_pnp_config.xml"));
    uart::SerialPort serial = uart::SerialPort(fmt::format("{}{}", CONFIG_FILE_PATH, "/uart_serial_config.xml"));

    while (mv_capture_.isindustryimgInput())
    {
        if (cv::waitKey(1) == 'q')
            break;
        mv_capture_.cameraReleasebuff();
        serial.updateReceiveInformation();
        src_img_ = mv_capture_.image();
        if (!src_img_.empty() && std::string(argv[1]) == "-d")
            cv::imshow("img", src_img_);
        auto res = yolov5_v5_Rtx_start(src_img_);

        cv::line(src_img_, cv::Point(0, src_img_.rows / 2), cv::Point(src_img_.cols, src_img_.rows / 2), cv::Scalar(0, 0, 0xFF));
        cv::line(src_img_, cv::Point(src_img_.cols / 2, 0), cv::Point(src_img_.cols / 2, src_img_.rows), cv::Scalar(0, 0, 0xFF));

        if (res.empty())
        {
            serial.updataWriteData(pnp.returnYawAngle(), pnp.returnPitchAngle(), pnp.returnDepth(), 0, 0);
            fmt::print("res empty.\n");
            continue;
        }

        cv::Rect rect = FilterRect(res, src_img_);
        if (rect.empty())
        {
            serial.updataWriteData(pnp.returnYawAngle(), pnp.returnPitchAngle(), pnp.returnDepth(), 0, 0);
            fmt::print("res after filter is empty.\n");
            continue;
        }

        pnp.solvePnP(30, 0, rect);
        serial.updataWriteData(pnp.returnYawAngle(), pnp.returnPitchAngle(), pnp.returnDepth(), 1, 0);

        if (std::string(argv[1]) == "-d")
        {
            cv::rectangle(src_img_, rect, cv::Scalar(0x27, 0xC1, 0x36), 2);
            cv::putText(src_img_, std::to_string(pnp.returnDepth()), cv::Point(rect.x, rect.y - 1), cv::FONT_HERSHEY_PLAIN, 1.2, cv::Scalar(0xFF, 0xFF, 0xFF), 2);
            fmt::format("rect x:{}, rect y:{} \n", rect.x, rect.y);
            fmt::format("rect width:{}, rect height:{} \n", rect.width, rect.height);
            fmt::format("yaw:{}, pitch:{} \n", pnp.returnYawAngle(), pnp.returnPitchAngle());
            cv::imshow("img", src_img_);
        }
    }
    return 0;
}