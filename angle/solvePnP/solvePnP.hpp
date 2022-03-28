#pragma once

#include <opencv2/opencv.hpp>
#include <fmt/color.h>
#include <fmt/core.h>
#include <iostream>

namespace solvepnp
{
    auto idntifier_green = fmt::format(fg(fmt::color::green), "solvePnP");
    auto idntifier_red = fmt::format(fg(fmt::color::red), "solvePnP");

    class PnP
    {
    public:
        PnP(std::string _camera_path,
            std::string _pnp_config_path)
        {
            cv::FileStorage fs_camera(_camera_path, cv::FileStorage::READ);
            fs_camera["camera-matrix"] >> cameraMatrix_;
            fs_camera["distortion"] >> distCoeffs_;

            cv::FileStorage fs_config(_pnp_config_path, cv::FileStorage::READ);
            fs_config["PTZ_CAMERA_X"] >> PnP_Config.ptz_camera_x;
            fs_config["PTZ_CAMERA_Y"] >> PnP_Config.ptz_camera_y;
            fs_config["PTZ_CAMERA_Z"] >> PnP_Config.ptz_camera_z;
            fs_config["PTZ_BARREL_X"] >> PnP_Config.barrel_ptz_offset_x;
            fs_config["PTZ_BARREL_Y"] >> PnP_Config.barrel_ptz_offset_y;
            fs_config["OFFSET_YAW"] >> PnP_Config.offset_yaw;
            fs_config["OFFSET_PITCH"] >> PnP_Config.offset_pitch;
        }

        /**
         * @brief 转换坐标系
         *
         * @param _t       旋转向量
         * @return cv::Mat 返回转化后的旋转向量
         * @author XX
         */
        cv::Mat cameraPtz(cv::Mat &_t)
        {
            static double theta = 0;
            static double r_data[] = {1, 0, 0,
                                      0, cos(theta), sin(theta),
                                      0, -sin(theta), cos(theta)};
            static double t_data[] = {static_cast<double>(PnP_Config.ptz_camera_x),
                                      static_cast<double>(PnP_Config.ptz_camera_y),
                                      static_cast<double>(PnP_Config.ptz_camera_z)};

            cv::Mat r_camera_ptz = cv::Mat(3, 3, CV_64FC1, r_data);
            cv::Mat t_camera_ptz = cv::Mat(3, 1, CV_64FC1, t_data);
            return r_camera_ptz * _t - t_camera_ptz;
        }

        /**
         * @brief PnP 解算
         *
         * @param object_3d_ 目标的实际坐标
         * @param target_2d_ 目标的图像坐标
         * @param angle Pitch Yaw
         * @param depth 深度
         */
        void solvePnP(const std::vector<cv::Point3f> object_3d_,
                      const std::vector<cv::Point2f> target_2d_,
                      cv::Point2f &angle,
                      cv::Point3f &coordinate_mm,
                      float &depth)
        {
            cv::Mat rvec_ = cv::Mat::zeros(3, 3, CV_64FC1);
            cv::Mat tvec_ = cv::Mat::zeros(3, 1, CV_64FC1);
            cv::solvePnP(object_3d_,
                         target_2d_,
                         cameraMatrix_,
                         distCoeffs_,
                         rvec_,
                         tvec_,
                         false,
                         cv::SOLVEPNP_SQPNP);
            float pitch_angle = 0.f;
            float yaw_angle = 0.f;

            cv::Mat ptz = cameraPtz(tvec_);
            const double *xyz = reinterpret_cast<const double *>(ptz.data);
            coordinate_mm.x = xyz[0];
            coordinate_mm.y = xyz[1];
            coordinate_mm.z = xyz[2];

            // Yaw
            if (PnP_Config.barrel_ptz_offset_x != 0.f)
            {
                double alpha =
                    asin(static_cast<double>(PnP_Config.barrel_ptz_offset_x) /
                         sqrt(xyz[0] * xyz[0] + xyz[2] * xyz[2]));
                double beta = 0.f;

                if (xyz[0] > 0)
                {
                    beta = atan(-xyz[0] / xyz[2]);
                    yaw_angle = static_cast<float>(-(alpha + beta)); // camera coordinate
                }
                else if (xyz[0] < static_cast<double>(PnP_Config.barrel_ptz_offset_x))
                {
                    beta = atan(xyz[0] / xyz[2]);
                    yaw_angle = static_cast<float>(-(alpha - beta));
                }
                else
                {
                    beta = atan(xyz[0] / xyz[2]);
                    yaw_angle = static_cast<float>(beta - alpha); // camera coordinate
                }
            }
            else
            {
                yaw_angle = static_cast<float>(atan2(xyz[0], xyz[2]));
            }
            yaw_angle  = static_cast<float>(yaw_angle) * 180 / CV_PI;
            yaw_angle += PnP_Config.offset_yaw;
            angle.y = yaw_angle;

            // Pitch
            if (PnP_Config.barrel_ptz_offset_y != 0.f)
            {
                double alpha =
                    asin(static_cast<double>(PnP_Config.barrel_ptz_offset_y) /
                         sqrt(xyz[1] * xyz[1] + xyz[2] * xyz[2]));
                double beta = 0.f;
                if (xyz[1] < 0)
                {
                    beta = atan(-xyz[1] / xyz[2]);
                    pitch_angle = static_cast<float>(-(alpha + beta)); // camera coordinate
                }
                else if (xyz[1] < static_cast<double>(PnP_Config.barrel_ptz_offset_y))
                {
                    beta = atan(xyz[1] / xyz[2]);
                    pitch_angle = static_cast<float>(-(alpha - beta));
                }
                else
                {
                    beta = atan(xyz[1] / xyz[2]);
                    pitch_angle = static_cast<float>((beta - alpha)); // camera coordinate
                }
            }
            else
            {
                pitch_angle = static_cast<float>(atan2(xyz[1], xyz[2]));
            }
            pitch_angle = static_cast<float>(pitch_angle) * 180 / CV_PI;
            pitch_angle += PnP_Config.offset_pitch;
            angle.x = pitch_angle;

            // Depth
            depth = static_cast<float>(sqrt(xyz[2] * xyz[2] + xyz[0] * xyz[0]));
        }

        void solvePnP(const cv::Rect object_3d_rect,
                      const cv::Rect target_2d_rect_,
                      cv::Point2f &angle,
                      cv::Point3f &coordinate_mm,
                      float &depth)
        {
            std::vector<cv::Point3f> object_3d_;
            object_3d_.emplace_back(
                cv::Point3f(-object_3d_rect.width * 0.5,
                            -object_3d_rect.height * 0.5, 0));
            object_3d_.emplace_back(
                cv::Point3f(object_3d_rect.width * 0.5,
                            -object_3d_rect.height * 0.5, 0));
            object_3d_.emplace_back(
                cv::Point3f(object_3d_rect.width * 0.5,
                            object_3d_rect.height * 0.5, 0));
            object_3d_.emplace_back(
                cv::Point3f(-object_3d_rect.width * 0.5,
                            object_3d_rect.height * 0.5, 0));

            std::vector<cv::Point2f> target_2d_;
            target_2d_.emplace_back(cv::Point2f(target_2d_rect_.x, target_2d_rect_.y + target_2d_rect_.height));
            target_2d_.emplace_back(cv::Point2f(target_2d_rect_.x + target_2d_rect_.width, target_2d_rect_.y + target_2d_rect_.height));
            target_2d_.emplace_back(cv::Point2f(target_2d_rect_.x + target_2d_rect_.width, target_2d_rect_.y));
            target_2d_.emplace_back(cv::Point2f(target_2d_rect_.x, target_2d_rect_.y));

            this->solvePnP(object_3d_, target_2d_, angle, coordinate_mm, depth);
        }

    private:
        struct PnPConfig
        {
            double ptz_camera_x = 0.0;       //相机与云台的 X 轴偏移(左负右正)
            double ptz_camera_y = 0.0;       //相机与云台的 Y 轴偏移(上负下正)
            double ptz_camera_z = 0.0;       //相机与云台的 Z 轴偏移(前正后负)
            float barrel_ptz_offset_x = 0.0; //云台与枪管的 X 轴偏移(左负右正)
            float barrel_ptz_offset_y = 0.0; //云台与枪管的 Y 轴偏移(上负下正)
            float offset_pitch = 0.0;        // yaw 轴固定补偿
            float offset_yaw = 0.0;          // pitch 轴固定补偿
        } PnP_Config;

        cv::Mat cameraMatrix_, distCoeffs_;
    };

}