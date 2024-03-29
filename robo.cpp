#include "robo.hpp"

using namespace std::chrono_literals;

RoboR1::RoboR1() try {
  std::ifstream config_is(fmt::format("{}{}", CONFIG_FILE_PATH,
                          "/robo_config.json"));
  config_is >> config_json;
  config_json["debug_mode"].get_to<bool>(debug_mode);

#ifdef USE_TRT_DETECTOR
  yolo_detection_ = std::make_unique<TRTDetector>(
    fmt::format("{}{}", SOURCE_PATH, "/models/ball.engine"));
#endif

#ifdef USE_OV_DETECTOR
  yolo_ov_detector_ = std::make_unique<Detector>(
    fmt::format("{}{}", SOURCE_PATH, "/models/ball.xml"),
    2, 0.45f);
  yolo_ov_detector_->Init();
#endif

  pnp_ = std::make_unique<solvepnp::PnP>(
    fmt::format("{}{}", CONFIG_FILE_PATH, "/camera_273.xml"),
    fmt::format("{}{}", CONFIG_FILE_PATH, "/pnp_config.xml"));
  kalman_prediction_ = std::make_unique<KalmanPrediction>();
  is_kalman_open_ = config_json["is_kalman_open"].get<bool>();
  camera_ = std::make_shared<mindvision::VideoCapture>(mindvision::RESOLUTION::RESOLUTION_1280_X_1024,
    fmt::format("{}{}", CONFIG_FILE_PATH, "/camera_param.yaml"));
  serial_ = std::make_unique<RoboSerial>(
    config_json["serial_port"].get<std::string>(), 115200);
  streamer_ = std::make_unique<RoboStreamer>();
  if(debug_mode)
    pj_udp_cl_ = std::make_unique<UDPClient>(
      config_json["pj_udp_cl_port"].get<int>(),
      config_json["pj_udp_cl_ip"].get<std::string>());
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

  while (!end_node_) {
    if (uartWriteThread.joinable())
      uartWriteThread.detach();
    if (uartReadThread.joinable())
      uartReadThread.detach();
    if (detectionThread.joinable())
      detectionThread.detach();
    if (camera_->isOpen())
      streamer_->call_html_js_function("HardwareState(\"camera_state\", true);");
    else
      streamer_->call_html_js_function("HardwareState(\"camera_state\", false);");
    if(serial_->isOpen())
      streamer_->call_html_js_function("HardwareState(\"serial_state\", true);");
    else
      streamer_->call_html_js_function("HardwareState(\"serial_state\", false);");
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
        std::lock_guard<std::mutex> lck(mtx);
        serial_->WriteInfo(robo_cmd);
      } else {
        serial_->open();
      }
      std::this_thread::sleep_for(10ms);
    } catch (const std::exception &e) {
      serial_->close();
      static int serial_read_excepted_times{0};
      if (serial_read_excepted_times++ > 3) {
        fmt::print("[{}] serial excepted to many times, sleep 3s.\n",
                   serial::idntifier_red);
        std::this_thread::sleep_for(3000ms);
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
  if(req.getTarget() == "/catchBall" && req.getMethod() == "POST") {
    RoboCatchBallUartBuff ub;
    ub.code = std::stoi(req.getBody());
    std::lock_guard<std::mutex> lck(mtx);
    std::this_thread::sleep_for(10ms);
    if(serial_->isOpen()) {
      ub.crc8x = serial::crc8x_cal((uint8_t *)&ub + 1,
                                   sizeof(ub) - 3);
      for(int i = 0; i < 3; i++) {
        serial_->write((uint8_t *)&ub, sizeof(ub));
        std::this_thread::sleep_for(10ms);
      }
    }
  }
  if(req.getTarget() == "/kalmanPrediction" && req.getMethod() == "POST") {
    if(req.getBody() == "close")
      is_kalman_open_ = false;
    else if(req.getBody() == "open")
      is_kalman_open_ = true;
  }
}

void RoboR1::detectionTask() {
  float pnp_yaw_factor = config_json["pnp_yaw_factor"].get<float>(); // pnp 修正倍率
  float imu_yaw{0.f};
  bool is_lose{true};
  // int lose_target_times{0};
  float depth{0};
  cv::Mat src_img;
  cv::Rect target_rect_3d(0, 0, 150, 150);
  cv::Rect first_detect_region((camera_->getImageCols() - 600) * 0.5,
                               (camera_->getImageRows() - 600) * 0.5,
                               600,
                               600);
  cv::Rect follow_detect_region(first_detect_region);
  cv::Point2f target_pnp_angle;
  std::vector<myrobo::detection> pred;
  float pitch_compensate{0.f};

  ThreadPool pool(4);

  while (!end_node_) try {
    cv::Rect target_rect;
    if(camera_->isOpen()) {
      *camera_ >> src_img;

      imu_yaw = (robo_inf.yaw_angle.load() + imu_yaw) / 2;
      cv::Mat detect_roi_region = src_img(follow_detect_region);
      cv::Mat detect_roi_region_clone = detect_roi_region.clone();

#ifdef USE_TRT_DETECTOR
      yolo_detection_->detect(detect_roi_region_clone, pred);
#endif
#ifdef USE_OV_DETECTOR
      yolo_ov_detector_->detect(detect_roi_region_clone, pred);
      // yolo_ov_detector_->detect(src_img, pred);
#endif

      for(auto &i : pred) {
        i.rect.x += follow_detect_region.x;
        i.rect.y += follow_detect_region.y;
      }
      targetFilter(pred, target_rect, is_lose);
    } else {
      is_lose = true;
      camera_->open();
      std::this_thread::sleep_for(1000ms);
    }

    cv::Point2f detection_pnp_angle;
    cv::Point3f target_pnp_coordinate_mm;

    if(is_lose == false) {
      // lose_target_times = 0;
      // follow_detect_region.x = target_rect.x + target_rect.width / 2
      //                         - follow_detect_region.width / 2;
      // follow_detect_region.y = target_rect.y + target_rect.height / 2
      //                         - follow_detect_region.height / 2;
      // if(follow_detect_region.x < 0)
      //   follow_detect_region.x = 0;
      // if(follow_detect_region.y < 0)
      //   follow_detect_region.y = 0;
      // if(follow_detect_region.x + follow_detect_region.width > src_img.cols)
      //   follow_detect_region.x = src_img.cols - follow_detect_region.width;
      // if(follow_detect_region.y + follow_detect_region.height > src_img.rows)
      //   follow_detect_region.y = src_img.rows - follow_detect_region.height;

      // float detection_pnp_yaw_previous = detection_pnp_angle.y;
      pnp_->solvePnP(target_rect_3d, target_rect, detection_pnp_angle,
                      target_pnp_coordinate_mm, depth);
      detection_pnp_angle.y *= pnp_yaw_factor;
      // detection_pnp_angle.y = (detection_pnp_yaw_previous + detection_pnp_angle.y) / 2;

      // kalman 预测
      // float kalman_yaw_compensate =
      //   kalman_prediction_->Prediction(imu_yaw - detection_pnp_angle.y, depth);
      // target_pnp_angle = detection_pnp_angle;
      // target_pnp_angle.y -= kalman_yaw_compensate * 13;

      // 弹道补偿
      // float depth_m = depth / 1000.f;
      // float depth_m = 8.1f;
      // pitch_compensate = 0.0132f * float(pow(depth_m, 4)) -
      //                          0.2624f * float(pow(depth_m, 3)) +
      //                          2.1572f * float(pow(depth_m, 2)) -
      //                          7.3033f * depth_m + 17.783f - 16.3f;

      // detection_pnp_angle.x -= pitch_compensate;
      // target_pnp_angle.x -= pitch_compensate;
    }

    // if(!robo_inf.following.load()) {
    //   follow_detect_region = first_detect_region;
    //   is_lose = true;
    // }

    if(is_lose == false) {
      if(is_kalman_open_) {
        storeRoboInfo(target_pnp_angle, depth, true);
      } else {
        storeRoboInfo(detection_pnp_angle, depth, true);
      }
    } else {
      // if(lose_target_times++ < 5) {
      //   storeRoboInfo(detection_pnp_angle, depth, true);
      // } else {
        cv::Point2f empty;
        detection_pnp_angle = empty;
        target_pnp_angle = empty;
        storeRoboInfo(empty, 0.f, false);
      // }
    }

    if (!src_img.empty()) {
      cv::Scalar line;
      if(!robo_inf.following.load())
        line = cv::Scalar(0, 150, 255);
      else
        line = cv::Scalar(0, 0, 255);
      // 准星
      cv::line(src_img, cv::Point(0, src_img.rows / 2),
               cv::Point(src_img.cols, src_img.rows / 2),
               line, 2);
      cv::line(src_img, cv::Point(src_img.cols / 2, 0),
               cv::Point(src_img.cols / 2, src_img.rows),
               line, 2);

      // 搜索范围
      cv::rectangle(src_img, follow_detect_region,
                    line, 5);

      // 跟踪目标
      for(auto &i : pred) {
        cv::rectangle(src_img, i.rect, cv::Scalar(0, 255, 0), 2);
      }
      cv::rectangle(src_img, target_rect, cv::Scalar(0, 150, 255), 5);
      cv::putText(src_img, std::to_string(depth),
        cv::Point(target_rect.x, target_rect.y - 1),
        cv::FONT_HERSHEY_DUPLEX, 1.5, cv::Scalar(255, 0, 150));

      pool.enqueue([=]() {
        if (debug_mode) {
          debug_info_["imu_yaw"] = robo_inf.yaw_angle.load();
          debug_info_["imu_yaw-vision_yaw"] = robo_inf.yaw_angle.load() - robo_cmd.yaw_angle.load();
          debug_info_["visoion_yaw"] = robo_cmd.yaw_angle.load();
          debug_info_["vision_pitch"] = robo_cmd.pitch_angle.load();
          debug_info_["is_lose"] = is_lose;
          pj_udp_cl_->send_message(debug_info_.dump());
          debug_info_.empty();

          cv::Mat resize_src_img;
          cv::resize(src_img, resize_src_img, cv::Size(), 0.25, 0.25);

          // 网页图传
          std::vector<uchar> buff_bgr;
          cv::imencode(".jpg", resize_src_img, buff_bgr);
          streamer_->publish("/pc",
                            std::string(buff_bgr.begin(),
                            buff_bgr.end()));
          streamer_->publish_text_value("yaw_angle", robo_cmd.yaw_angle.load());
          streamer_->publish_text_value("pitch_angle", robo_cmd.pitch_angle.load());
        }
      });

      // 击打指示
      // if(is_lose == false &&
      //    ((abs(detection_pnp_angle.y) < 0.2f && abs(target_pnp_angle.y) < 0.2f) ||
      //     (detection_pnp_angle.y < (target_pnp_angle.y - detection_pnp_angle.y) &&
      //      target_pnp_angle.y < 1.f) ||
      //     (detection_pnp_angle.y > (target_pnp_angle.y - detection_pnp_angle.y) &&
      //      target_pnp_angle.y > -1.f)))
      //   streamer_->call_html_js_function("HardwareState(\"ready_to_shoot\", false);");
      // else
      //   streamer_->call_html_js_function("HardwareState(\"ready_to_shoot\", true);");
    }
    std::this_thread::sleep_for(1ms);
  } catch (const std::exception &e) {
    fmt::print("{}\n", e.what());
  }
}

void RoboR1::targetFilter(std::vector<myrobo::detection> &_pred,
    cv::Rect &_target, bool &_is_lose) {
  // cv::Rect select_rect;
  if(_pred.size() > 0) {
    std::sort(_pred.begin(), _pred.end(),
      [](myrobo::detection _a, auto _b){return pow(_a.rect.x, 2) + pow(_a.rect.y, 2) < 
        pow(_b.rect.x, 2) + pow(_b.rect.y, 2);});
    _target = _pred[0].rect;
    _target.height = _target.width;
    _is_lose = false;
  } else {
    _is_lose = true;
  }
  _pred.clear();
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
