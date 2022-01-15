#include <opencv2/opencv.hpp>
#include <iostream>
#include <fmt/color.h>
#include <fmt/core.h>

#include "Camera/mv_video_capture.hpp"
#include "TensorRTx/yolov5.hpp"
#include "serial/uart_serial.hpp"
#include "angle/solvePnP/solvePnP.hpp"

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
                                                                   50000);
    mindvision::VideoCapture mv_capture_ = mindvision::VideoCapture(CameraParams);

    start(fmt::format("{}{}", SOURCE_PATH, "/models/RCBall3.engine"));

    cv::Mat src_img_;

    solvepnp::PnP pnp(fmt::format("{}{}", CONFIG_FILE_PATH, "/camera_273.xml"),
                      fmt::format("{}{}", CONFIG_FILE_PATH, "/pnp_config.xml"));
    uart::SerialPort serial = uart::SerialPort(fmt::format("{}{}", CONFIG_FILE_PATH, "/uart_serial_config.xml"));

    while (mv_capture_.isindustryimgInput())
    {
        if (cv::waitKey(1) == 'q')
            break;
        mv_capture_.cameraReleasebuff();
        serial.updateReceiveInformation();
        src_img_ = mv_capture_.image();
        if (!src_img_.empty())
            cv::imshow("img", src_img_);
        auto res = yolov5_v5_Rtx_start(src_img_);

        cv::line(src_img_, cv::Point(0, src_img_.rows / 2), cv::Point(src_img_.cols, src_img_.rows / 2), cv::Scalar(0, 0, 0xFF));
        cv::line(src_img_, cv::Point(src_img_.cols / 2, 0), cv::Point(src_img_.cols / 2, src_img_.rows), cv::Scalar(0, 0, 0xFF));

        if (res.empty())
        {
            // serial.updataWriteData(angle.y, angle.x, depth, 0, 0);
            fmt::print("res empty.\n");
            continue;
        }

        cv::Rect rect = FilterRect(res, src_img_);
        if (rect.empty())
        {
            // serial.updataWriteData(angle.y, angle.x, depth, 0, 0);
            fmt::print("res after filter is empty.\n");
            continue;
        }

        rect.height = rect.width;
        cv::Rect ball_3d_rect(0, 0, 165, 165);
        cv::Point2f angle;
        float depth;
        pnp.solvePnP(ball_3d_rect, rect, angle, depth);
        // serial.updataWriteData(angle.y, angle.x, depth, 1, 0);

        cv::rectangle(src_img_, rect, cv::Scalar(0x27, 0xC1, 0x36), 2);
        cv::putText(src_img_, std::to_string(depth), cv::Point(rect.x, rect.y - 1), cv::FONT_HERSHEY_PLAIN, 1.2, cv::Scalar(0xFF, 0xFF, 0xFF), 2);
        fmt::print("pitch:{}, yaw:{} \n", angle.x, angle.y);
        cv::imshow("img", src_img_);
        fmt::print("-------next--------\n");
    }
    return 0;
}