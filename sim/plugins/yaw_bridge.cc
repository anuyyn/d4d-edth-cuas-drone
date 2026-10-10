#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>

#include <cmath>
#include <cstring>
#include <mutex>
#include <random>
#include <vector>

#include <gazebo/common/common.hh>
#include <gazebo/gazebo.hh>
#include <gazebo/physics/physics.hh>
#include <gazebo/rendering/rendering.hh>
#include <gazebo/sensors/sensors.hh>
#include <ignition/math/Pose3.hh>

namespace gazebo {

#pragma pack(push, 1)
struct FrameHdr {
  uint32_t magic;
  uint32_t seq;
  double sim_t;
  float err_true;
  float dist;
  float yaw_rate;
  uint16_t w, h;
};
struct CmdMsg {
  uint32_t magic;
  uint32_t seq;
  float cmd;
  uint32_t reset;
  uint32_t seed;
};
#pragma pack(pop)
static const uint32_t MAGIC = 0x42574159;

class YawBridge : public ModelPlugin {
 public:
  ~YawBridge() override {
    if (sock_ >= 0) close(sock_);
  }

  void Load(physics::ModelPtr model, sdf::ElementPtr sdf) override {
    model_ = model;
    world_ = model->GetWorld();
    auto get = [&](const char* k, double d) { return sdf->HasElement(k) ? sdf->Get<double>(k) : d; };
    camName_ = sdf->HasElement("camera") ? sdf->Get<std::string>("camera") : "cam";
    balloonName_ = sdf->HasElement("balloon") ? sdf->Get<std::string>("balloon") : "balloon";
    fullRate_ = get("full_stick_rate_dps", 400.0) * M_PI / 180.0;
    tau_ = get("yaw_tau", 0.08);
    radius_ = get("balloon_radius", 6.0);
    speed_ = get("balloon_speed", 1.0);
    drift_ = get("pilot_drift", 0.05);
    timeout_ms_ = (int)get("timeout_ms", 3000);
    int portIn = (int)get("port_in", 9102), portOut = (int)get("port_out", 9101);

    sock_ = socket(AF_INET, SOCK_DGRAM, 0);
    sockaddr_in me{};
    me.sin_family = AF_INET;
    me.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    me.sin_port = htons(portIn);
    if (bind(sock_, (sockaddr*)&me, sizeof(me)) < 0) gzerr << "[yaw_bridge] bind " << portIn << " 실패\n";
    peer_ = {};
    peer_.sin_family = AF_INET;
    peer_.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    peer_.sin_port = htons(portOut);
    timeval tv{timeout_ms_ / 1000, (timeout_ms_ % 1000) * 1000};
    setsockopt(sock_, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));

    home_ = model_->WorldPose().Pos();
    NewEpisode(1);
    updConn_ = event::Events::ConnectWorldUpdateBegin(std::bind(&YawBridge::OnUpdate, this, std::placeholders::_1));
    gzmsg << "[yaw_bridge] loaded: in=" << portIn << " out=" << portOut << " full_rate=" << fullRate_ * 180 / M_PI
          << "dps tau=" << tau_ << "\n";
  }

 private:
  void NewEpisode(uint32_t seed) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<double> U(0, 2 * M_PI), A(0.6, 1.4);
    for (int i = 0; i < 3; ++i) { ph_[i] = U(rng); amp_[i] = A(rng); }
    az0_ = U(rng);
    yawRate_ = 0; cmd_ = 0; t0_ = world_->SimTime().Double();
    drng_.seed(seed + 7);
    pos_ = home_;
    auto b0 = BalloonPos(0) - home_;
    yaw_ = std::atan2(b0.Y(), b0.X()) + std::uniform_real_distribution<double>(-0.3, 0.3)(rng);
  }

  ignition::math::Vector3d BalloonPos(double t) const {
    double s = speed_;
    double az = az0_ + amp_[0] * 0.9 * std::sin(0.35 * s * t + ph_[0]) + amp_[1] * 0.5 * std::sin(0.9 * s * t + ph_[1]);
    double r = radius_ * (1.0 + 0.25 * std::sin(0.2 * s * t + ph_[2]));
    double z = home_.Z() + 0.8 * amp_[2] * std::sin(0.3 * s * t + ph_[1]);
    return {home_.X() + r * std::cos(az), home_.Y() + r * std::sin(az), z};
  }

  void TryAttachCamera() {
    if (camConn_) return;
    for (auto& s : sensors::SensorManager::Instance()->GetSensors()) {
      if (s->Name() != camName_) continue;
      auto cam = std::dynamic_pointer_cast<sensors::CameraSensor>(s);
      if (!cam || !cam->Camera()) return;
      cam_ = cam;
      camConn_ = cam->Camera()->ConnectNewImageFrame(
          [this](const unsigned char* img, unsigned w, unsigned h, unsigned, const std::string&) {
            std::lock_guard<std::mutex> lk(mtx_);
            frame_.assign(img, img + w * h * 3);
            fw_ = w; fh_ = h; ready_ = true;
          });
      gzmsg << "[yaw_bridge] camera '" << camName_ << "' attached\n";
    }
  }

  static double Wrap(double a) { return std::atan2(std::sin(a), std::cos(a)); }

  void OnUpdate(const common::UpdateInfo& info) {
    TryAttachCamera();
    double t = info.simTime.Double() - t0_;
    double dt = lastT_ < 0 ? 0 : info.simTime.Double() - lastT_;
    lastT_ = info.simTime.Double();

    auto bp = BalloonPos(t);
    if (auto b = world_->ModelByName(balloonName_)) b->SetWorldPose({bp, {0, 0, 0}});

    std::vector<uint8_t> img;
    unsigned w = 0, h = 0;
    {
      std::lock_guard<std::mutex> lk(mtx_);
      if (ready_) { img.swap(frame_); w = fw_; h = fh_; ready_ = false; }
    }
    if (!img.empty()) Exchange(img, w, h, t, bp);


    double target = -cmd_ * fullRate_;
    if (dt > 0) {
      yawRate_ += (target - yawRate_) * std::min(1.0, dt / tau_);
      yaw_ = Wrap(yaw_ + yawRate_ * dt);
      std::normal_distribution<double> N(0, drift_ * std::sqrt(dt));
      pos_ += {N(drng_), N(drng_), 0.5 * N(drng_)};
      pos_ = home_ + (pos_ - home_) * (1 - 0.2 * dt);
    }
    model_->SetWorldPose({pos_, {0, 0, yaw_}});
  }

  void Exchange(const std::vector<uint8_t>& img, unsigned w, unsigned h, double t,
                const ignition::math::Vector3d& bp) {
    auto d = bp - pos_;
    double rel = Wrap(std::atan2(d.Y(), d.X()) - yaw_);
    FrameHdr hdr{MAGIC, ++seq_, t, (float)-rel, (float)d.Length(), (float)-yawRate_, (uint16_t)w, (uint16_t)h};
    std::vector<uint8_t> pkt(sizeof(hdr) + img.size());
    std::memcpy(pkt.data(), &hdr, sizeof(hdr));
    std::memcpy(pkt.data() + sizeof(hdr), img.data(), img.size());
    sendto(sock_, pkt.data(), pkt.size(), 0, (sockaddr*)&peer_, sizeof(peer_));


    CmdMsg m{};
    while (true) {
      ssize_t n = recv(sock_, &m, sizeof(m), 0);
      if (n < 0) { cmd_ = 0; return; }
      if (n == sizeof(m) && m.magic == MAGIC && m.seq == seq_) break;
    }
    cmd_ = std::max(-1.0f, std::min(1.0f, m.cmd));
    if (m.reset) NewEpisode(m.seed);
  }

  physics::ModelPtr model_;
  physics::WorldPtr world_;
  sensors::CameraSensorPtr cam_;
  event::ConnectionPtr updConn_, camConn_;
  std::string camName_, balloonName_;
  int sock_ = -1, timeout_ms_ = 3000;
  sockaddr_in peer_{};
  std::mutex mtx_;
  std::vector<uint8_t> frame_;
  unsigned fw_ = 0, fh_ = 0;
  bool ready_ = false;
  uint32_t seq_ = 0;
  double fullRate_, tau_, radius_, speed_, drift_;
  double yaw_ = 0, yawRate_ = 0, cmd_ = 0, t0_ = 0, lastT_ = -1, az0_ = 0;
  double ph_[3]{}, amp_[3]{};
  ignition::math::Vector3d home_, pos_;
  std::mt19937 drng_;
};

GZ_REGISTER_MODEL_PLUGIN(YawBridge)
}
