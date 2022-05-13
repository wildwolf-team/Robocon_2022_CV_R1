#include "robo.hpp"

using namespace std::chrono_literals;

namespace myrobo {
  struct ROSInfoPub{
    std::shared_ptr<rclcpp::Publisher<std_msgs::msg::Float32>> yaw_detect_publisher;
    std::shared_ptr<rclcpp::Publisher<std_msgs::msg::Float32>> pitch_detect_publisher;
    std::shared_ptr<rclcpp::Publisher<std_msgs::msg::Float32>> kalman_in_publisher;
    std::shared_ptr<rclcpp::Publisher<std_msgs::msg::Float32>> imu_yaw_publisher;

    void nh_init(rclcpp::Node::SharedPtr n_){
      yaw_detect_publisher =
        n_->create_publisher<std_msgs::msg::Float32>("ptz/detect/yaw",
          rclcpp::QoS(rclcpp::KeepAll()));
      pitch_detect_publisher =
        n_->create_publisher<std_msgs::msg::Float32>("ptz/detect/pitch",
          rclcpp::QoS(rclcpp::KeepAll()));
      kalman_in_publisher =
        n_->create_publisher<std_msgs::msg::Float32>("ptz/detect/kalman_in",
          rclcpp::QoS(rclcpp::KeepAll()));
      imu_yaw_publisher =
        n_->create_publisher<std_msgs::msg::Float32>("ptz/detect/imu_yaw",
          rclcpp::QoS(rclcpp::KeepAll()));
    }
  };

  struct detection {
    cv::Rect rect;
    float class_id;
    float conf;
  };
}

RoboR1::RoboR1() try {
  n_ = rclcpp::Node::make_shared("robo_r1_node");
  yolo_detection_ = std::make_unique<YOLOv5TRT>(
    fmt::format("{}{}", SOURCE_PATH, "/models/ball.engine"));
  pnp_ = std::make_unique<solvepnp::PnP>(
    fmt::format("{}{}", CONFIG_FILE_PATH, "/camera_273.xml"),
    fmt::format("{}{}", CONFIG_FILE_PATH, "/pnp_config.xml"));
  kalman_prediction_ = std::make_unique<KalmanPrediction>();

  int camera_exposure = 5000;
  mindvision::CameraParam camera_params(0, mindvision::RESOLUTION_1280_X_1024,
                                          camera_exposure);
  camera_ = std::make_shared<mindvision::VideoCapture>(camera_params);
  serial_ = std::make_unique<RoboSerial>("/dev/ttyACM0", 115200);
  streamer_ = std::make_unique<RoboStreamer>();
  ros_pub_.nh_init(n_);
} catch (const std::exception &e) {
  fmt::print("[{}] {}\n", fmt::format(fg(fmt::color::red) |
              fmt::emphasis::bold, "construct"), e.what());
}

void RoboR1::Init() try {
  streamer_->setCallbackFuncPtr(
    std::bind(&RoboR1::streamerCallback, this, std::placeholders::_1));

  if(!streamer_->isRunning())
    streamer_->start(8080, SOURCE_PATH "/streamer/streamer.html");
  if(!camera_->isOpen())
    camera_->open();
} catch (const std::exception &e) {
  fmt::print("[{}] {}\n", fmt::format(fg(fmt::color::red) |
              fmt::emphasis::bold, "init"), e.what());
}

void RoboR1::Spin() {
  std::thread uartWriteThread(std::bind(&RoboR1::uartWrite,this));
  std::thread uartReadThread(std::bind(&RoboR1::uartRead,this));
  std::thread detectionThread(std::bind(&RoboR1::detectionTask,this));

  while (!end_node_ && rclcpp::ok()) {
    if (uartWriteThread.joinable())
      uartWriteThread.detach();
    if (uartReadThread.joinable())
      uartReadThread.detach();
    if (detectionThread.joinable())
      detectionThread.detach();
    std::this_thread::sleep_for(1000ms);
  }

  if(!end_node_){
    end_node_ = true;
    sleep(1);
  }

  try {
    if(serial_->isOpen())
      serial_->close();
    if(streamer_->isRunning())
      streamer_->stop();
    rclcpp::shutdown();
  } catch (const std::exception& e) {
    fmt::print("{}\n", e.what());
  }
}

void RoboR1::uartRead() {
  while (!end_node_) try {
      if(serial_->isOpen()) {
        serial_->ReceiveInfo(robo_inf);
        streamer_->publish_text_value("imu_angle",
                                      robo_inf.yaw_angle.load());
      }
    } catch (const std::exception &e) {
      std::this_thread::sleep_for(1000ms);
    }
}

void RoboR1::uartWrite() {
  while (!end_node_) try {
      if(serial_->isOpen()) {
        serial_->WriteInfo(robo_cmd);
      } else {
        serial_->open();
      }
      std::this_thread::sleep_for(10ms);
    } catch (const std::exception &e) {
      streamer_->call_html_js_function("HardwareState(\"serial_state\", false);");
      serial_->close();
      static int serial_read_excepted_times{0};
      if (serial_read_excepted_times++ > 3) {
        fmt::print("[{}] serial excepted to many times, sleep 10s.\n",
                   serial::idntifier_red);
        std::this_thread::sleep_for(10000ms);
        serial_read_excepted_times = 0;
      }
      fmt::print("[{}] {}\n",
                 serial::idntifier_red, e.what());
      std::this_thread::sleep_for(1000ms);
    }
}

void RoboR1::streamerCallback(const nadjieb::net::HTTPRequest &req) {
  if (req.getTarget() == "/stop") {
    stop();
  }
  if ((req.getTarget() == "/setCameraExposure") && (req.getMethod() == "POST")){
    camera_->setCameraExposureTime(stoi(req.getBody()));
  }
  if ((req.getTarget() == "/setCameraOnceWB")){
    camera_->setCameraOnceWB();
  }
  if (req.getTarget() == "/gamepadInput" && req.getMethod() == "POST") {
    std::map<std::string, float> gamepad_state_key_val;
    std::string gamepad_input_str = req.getBody();
    std::istringstream iss(gamepad_input_str);
    std::string key, val;
    while (true) {
      key="";
      std::getline(iss, key, '=');
      if(key == "") break;
      std::getline(iss, val, '&');
      gamepad_state_key_val[key] = std::stof(val);
    }
  }
}

void RoboR1::detectionTask() {
  float pnp_yaw_factor{1.05f}; // pnp 修正倍率
  bool is_lose{true};
  int lose_target_times{0};
  float depth{0};
  cv::Mat src_img;
  cv::Rect target_rect;
  cv::Rect target_rect_predicted;
  cv::Rect target_rect_3d(0, 0, 150, 150);
  cv::Rect first_detect_region(camera_->getImageCols() / 4,
                               camera_->getImageRows() / 2,
                               camera_->getImageCols() / 2,
                               camera_->getImageRows() / 3);
  cv::Rect follow_detect_region(first_detect_region);
  cv::Point2f detection_pnp_angle;
  cv::Point2f target_pnp_angle;
  cv::Point3f target_pnp_coordinate_mm;
  std::vector<myrobo::detection> pred;

  while (!end_node_) try {
    if(camera_->isOpen()) {
      *camera_ >> src_img;
      cv::Mat detect_roi_region = src_img(follow_detect_region);
      cv::Mat detect_roi_region_clone = detect_roi_region.clone();
      auto res = yolo_detection_->Detect(detect_roi_region_clone);
      yoloDetectionCovert(res, detect_roi_region_clone, pred);
      for(auto &i : pred) {
        i.rect.x += follow_detect_region.x;
        i.rect.y += follow_detect_region.y;
      }
      targetFilter(pred, follow_detect_region, target_rect, is_lose);
    } else {
      streamer_->call_html_js_function("HardwareState(\"camera_state\", false);");
      is_lose = true;
      camera_->open();
      std::this_thread::sleep_for(1000ms);
    }

    if(robo_inf.following.load() && is_lose == false) {
      lose_target_times = 0;
      follow_detect_region.x = target_rect.x + target_rect.width / 2
                              - follow_detect_region.width / 2;
      follow_detect_region.y = target_rect.y + target_rect.height / 2
                              - follow_detect_region.height / 2;
      if(follow_detect_region.x < 0)
        follow_detect_region.x = 0;
      if(follow_detect_region.y < 0)
        follow_detect_region.y = 0;
      if(follow_detect_region.x + follow_detect_region.width > src_img.cols)
        follow_detect_region.x = src_img.cols - follow_detect_region.width;
      if(follow_detect_region.y + follow_detect_region.height > src_img.rows)
        follow_detect_region.y = src_img.rows - follow_detect_region.height;

      pnp_->solvePnP(target_rect_3d, target_rect, detection_pnp_angle,
                      target_pnp_coordinate_mm, depth);
      detection_pnp_angle.y *= pnp_yaw_factor;

      // kalman 预测
      float kalman_yaw_compensate =
        kalman_prediction_->Prediction(robo_inf.yaw_angle.load() -
                                       detection_pnp_angle.y, depth, 15.f);
      target_rect_predicted = target_rect;
      target_rect_predicted.x = target_rect.x - kalman_yaw_compensate;
      pnp_->solvePnP(target_rect_3d, target_rect_predicted,
                    target_pnp_angle, target_pnp_coordinate_mm,
                    depth);
      target_pnp_angle.y *= pnp_yaw_factor;

      // 弹道补偿
      float depth_m = depth / 1000;
      float pitch_compensate = -0.0294f * float(pow(depth_m, 6)) +
                                0.8118f * float(pow(depth_m, 5)) -
                                9.2449f * float(pow(depth_m, 4)) +
                                55.925f * float(pow(depth_m, 3)) -
                                190.32f * float(pow(depth_m, 2)) +
                                347.f * depth_m - 257.12f;
      target_pnp_angle.x -= pitch_compensate;
    }

    if(!robo_inf.following.load()) {
      follow_detect_region = first_detect_region;
      is_lose = true;
    }

    if(is_lose == false && lose_target_times++ < 5) {
      if(is_kalman_open_) {
        storeRoboInfo(target_pnp_angle, depth, true);
      } else {
        storeRoboInfo(detection_pnp_angle, depth, true);
      }
    } else {
      cv::Point2f empty;
      storeRoboInfo(empty, 0.f, false);
    }

    if (!src_img.empty()) {
      // 准星
      cv::line(src_img, cv::Point(0, src_img.rows / 2),
               cv::Point(src_img.cols, src_img.rows / 2),
               cv::Scalar(0, 150, 255), 2);
      cv::line(src_img, cv::Point(src_img.cols / 2, 0),
               cv::Point(src_img.cols / 2, src_img.rows),
               cv::Scalar(0, 150, 255), 2);

      // 搜索范围
      if(!robo_inf.following.load())
        cv::rectangle(src_img, follow_detect_region,
                      cv::Scalar(0, 150, 255), 5);
      else
        cv::rectangle(src_img, follow_detect_region,
                      cv::Scalar(0, 0, 255), 5);

      // 跟踪目标
      for(auto &i : pred) {
        cv::rectangle(src_img, i.rect, cv::Scalar(0, 255, 0), 2);
      }
      cv::rectangle(src_img, target_rect, cv::Scalar(0, 150, 255), 5);
      cv::rectangle(src_img, target_rect_predicted,
                    cv::Scalar(255, 0, 150), 5);

      // 网页图传
      std::vector<uchar> buff_bgr;
      cv::imencode(".jpg", src_img, buff_bgr);
      streamer_->publish("/pc",
                        std::string(buff_bgr.begin(),
                        buff_bgr.end()));
      streamer_->publish_text_value("yaw_angle", target_pnp_angle.y);
      streamer_->publish_text_value("pitch_angle", target_pnp_angle.x);

      // 击打指示
      if(is_lose == false &&
         ((abs(detection_pnp_angle.y) < 1.f &&
         abs(target_pnp_angle.y) < 1.f) ||
         (detection_pnp_angle.y < -0.5f &&
         target_pnp_angle.y > 0.5f) ||
         (detection_pnp_angle.y > 0.5f &&
         target_pnp_angle.y < -0.5f)))
        streamer_->call_html_js_function("HardwareState(\"ready_to_shoot\", false);");
      else
        streamer_->call_html_js_function("HardwareState(\"ready_to_shoot\", true);");
    }
    usleep(1);
  } catch (const std::exception &e) {
    fmt::print("{}\n", e.what());
  }
}

void RoboR1::yoloDetectionCovert(std::vector<Yolo::Detection> &_res,
  cv::Mat &_img, std::vector<myrobo::detection> &_pred) {
  _pred.clear();
  for(auto &i : _res) {
    myrobo::detection dt;
    dt.rect = get_rect(_img, i.bbox);
    dt.class_id = i.class_id;
    dt.conf = i.conf;
    _pred.emplace_back(dt);
  }
}

void RoboR1::targetFilter(const std::vector<myrobo::detection> &_pred,
  const cv::Rect &_region, cv::Rect &_target, bool &_is_lose) {
  cv::Rect select_rect;
  if(_pred.size() > 0)
    select_rect = _pred[0].rect;
  for(auto& i : _pred) {
    if(i.conf < 0.3)
      continue;
    if(i.rect.width / i.rect.height > 1.2f ||
       i.rect.width / i.rect.height < 0.8f)
      continue;
    if((pow(abs(i.rect.x - _region.x + _region.width / 2), 2) + 
       pow(abs(i.rect.y - _region.y + _region.height / 2), 2)) <
       (pow(abs(select_rect.x - _region.x + _region.width / 2), 2) + 
       pow(abs(select_rect.y - _region.y + _region.height / 2), 2)))
      select_rect = i.rect;
  }
  _target = select_rect;
  select_rect.height = select_rect.width;
  if(_target.width == 0)
    _is_lose = true;
  else
    _is_lose = false;
}

void RoboR1::storeRoboInfo(const cv::Point2f &_pnp_angle,
  const float &_depth, const bool &_detect_object) {
  robo_cmd.pitch_angle.store(_pnp_angle.x);
  robo_cmd.yaw_angle.store(_pnp_angle.y);
  robo_cmd.depth.store(_depth);
  robo_cmd.detect_object.store(_detect_object);
}

void RoboR1::stop(){
  if (!end_node_) {
    end_node_ = true;
  }
  fmt::print("node stop.\n");
}

RoboR1::~RoboR1() {}
