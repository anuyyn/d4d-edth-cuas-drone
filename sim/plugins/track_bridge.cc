#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>

#include <array>
#include <cmath>
#include <cstring>
#include <mutex>
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
  uint32_t magic, seq;
  double sim_t;
  float pose[6];
  uint16_t w, h;
};
struct CmdMsg {
  uint32_t magic, seq;
  float pose[6];
  uint32_t reset;
};
struct ChaseHdr { uint32_t magic, seq; uint16_t idx, n, w, h; uint32_t off, len; };
#pragma pack(pop)
static const uint32_t MAGIC = 0x4B435254;
static const uint32_t MAGIC_CHASE = 0x45534843;
static const uint32_t MAGIC_IMG = 0x43474D49;

class TrackBridge : public ModelPlugin {
 public:
  ~TrackBridge() override { if (sock_ >= 0) close(sock_); }

  void Load(physics::ModelPtr model, sdf::ElementPtr sdf) override {
    model_ = model;
    world_ = model->GetWorld();
    camName_ = sdf->HasElement("camera") ? sdf->Get<std::string>("camera") : "cam";
    chaseName_ = sdf->HasElement("chase_camera") ? sdf->Get<std::string>("chase_camera") : "chase_cam";
    chaseModel_ = sdf->HasElement("chase_model") ? sdf->Get<std::string>("chase_model") : "chasecam";
    int portIn = sdf->HasElement("port_in") ? sdf->Get<int>("port_in") : 9302;
    int portOut = sdf->HasElement("port_out") ? sdf->Get<int>("port_out") : 9301;
    timeoutMs_ = sdf->HasElement("timeout_ms") ? sdf->Get<int>("timeout_ms") : 5000;
    auto p = model->WorldPose();
    pose_ = {(float)p.Pos().X(), (float)p.Pos().Y(), (float)p.Pos().Z(),
             (float)p.Rot().Roll(), (float)p.Rot().Pitch(), (float)p.Rot().Yaw()};

    sock_ = socket(AF_INET, SOCK_DGRAM, 0);
    sockaddr_in me{};
    me.sin_family = AF_INET;
    me.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    me.sin_port = htons(portIn);
    if (bind(sock_, (sockaddr*)&me, sizeof(me)) < 0) gzerr << "[track] bind " << portIn << " 실패\n";
    peer_.sin_family = AF_INET;
    peer_.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    peer_.sin_port = htons(portOut);
    timeval tv{timeoutMs_ / 1000, (timeoutMs_ % 1000) * 1000};
    setsockopt(sock_, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));
    updConn_ = event::Events::ConnectWorldUpdateBegin(std::bind(&TrackBridge::OnUpdate, this, std::placeholders::_1));
    gzmsg << "[track] loaded in=" << portIn << " out=" << portOut << "\n";
  }

 private:
  void Attach(const std::string& name, event::ConnectionPtr& conn, bool main) {
    if (conn) return;
    for (auto& s : sensors::SensorManager::Instance()->GetSensors()) {
      if (s->Name() != name) continue;
      auto cam = std::dynamic_pointer_cast<sensors::CameraSensor>(s);
      if (!cam || !cam->Camera()) return;
      conn = cam->Camera()->ConnectNewImageFrame(
          [this, main](const unsigned char* img, unsigned w, unsigned h, unsigned, const std::string&) {
            std::lock_guard<std::mutex> lk(mtx_);
            if (main) { frame_.assign(img, img + w * h * 3); fw_ = w; fh_ = h; ready_ = true; }
            else { chaseFrame_.assign(img, img + w * h * 3); cw_ = w; ch_ = h; }
          });
      gzmsg << "[track] camera attached: " << name << "\n";
    }
  }

  void OnUpdate(const common::UpdateInfo& info) {
    Attach(camName_, camConn_, true);
    Attach(chaseName_, chaseConn_, false);
    std::vector<uint8_t> img;
    unsigned w = 0, h = 0;
    {
      std::lock_guard<std::mutex> lk(mtx_);
      if (ready_) { img.swap(frame_); w = fw_; h = fh_; ready_ = false; }
    }
    if (!img.empty()) Exchange(img, w, h, info.simTime.Double());
    model_->SetWorldPose({pose_[0], pose_[1], pose_[2], pose_[3], pose_[4], pose_[5]});
    if (auto c = world_->ModelByName(chaseModel_)) {
      double yaw = pose_[5];
      chaseYaw_ += std::atan2(std::sin(yaw - chaseYaw_), std::cos(yaw - chaseYaw_)) * 0.05;
      ignition::math::Vector3d dp(pose_[0], pose_[1], pose_[2]);
      ignition::math::Vector3d cp = dp + ignition::math::Vector3d(-14 * std::cos(chaseYaw_), -14 * std::sin(chaseYaw_), 7);
      auto d = dp - cp;
      c->SetWorldPose({cp, {0, std::atan2(-d.Z(), std::hypot(d.X(), d.Y())), std::atan2(d.Y(), d.X())}});
    }
  }

  void Exchange(const std::vector<uint8_t>& img, unsigned w, unsigned h, double t) {
    FrameHdr hd{};
    hd.magic = MAGIC; hd.seq = ++seq_; hd.sim_t = t;
    std::memcpy(hd.pose, pose_.data(), sizeof(hd.pose));
    hd.w = w; hd.h = h;
    SendChase();
    SendChunks(MAGIC_IMG, img, w, h);
    sendto(sock_, &hd, sizeof(hd), 0, (sockaddr*)&peer_, sizeof(peer_));
    CmdMsg m{};
    while (true) {
      ssize_t n = recv(sock_, &m, sizeof(m), 0);
      if (n < 0) return;
      if (n == sizeof(m) && m.magic == MAGIC && m.seq == seq_) break;
    }
    std::memcpy(pose_.data(), m.pose, sizeof(m.pose));
  }

  void SendChase() {
    std::vector<uint8_t> f;
    unsigned w, h;
    {
      std::lock_guard<std::mutex> lk(mtx_);
      if (chaseFrame_.empty()) return;
      f = chaseFrame_; w = cw_; h = ch_;
    }
    SendChunks(MAGIC_CHASE, f, w, h);
  }

  void SendChunks(uint32_t magic, const std::vector<uint8_t>& f, unsigned w, unsigned h) {
    const uint32_t CH = 60000;
    uint16_t n = (f.size() + CH - 1) / CH;
    for (uint16_t i = 0; i < n; ++i) {
      uint32_t off = i * CH, len = std::min<uint32_t>(CH, f.size() - off);
      ChaseHdr c{magic, seq_, i, n, (uint16_t)w, (uint16_t)h, off, len};
      std::vector<uint8_t> pkt(sizeof(c) + len);
      std::memcpy(pkt.data(), &c, sizeof(c));
      std::memcpy(pkt.data() + sizeof(c), f.data() + off, len);
      sendto(sock_, pkt.data(), pkt.size(), 0, (sockaddr*)&peer_, sizeof(peer_));
      usleep(400);
    }
  }

  physics::ModelPtr model_;
  physics::WorldPtr world_;
  event::ConnectionPtr updConn_, camConn_, chaseConn_;
  std::string camName_, chaseName_, chaseModel_;
  int sock_ = -1, timeoutMs_ = 5000;
  sockaddr_in peer_{};
  std::mutex mtx_;
  std::vector<uint8_t> frame_, chaseFrame_;
  unsigned fw_ = 0, fh_ = 0, cw_ = 0, ch_ = 0;
  bool ready_ = false;
  uint32_t seq_ = 0;
  std::array<float, 6> pose_{};
  double chaseYaw_ = 0;
};

GZ_REGISTER_MODEL_PLUGIN(TrackBridge)
}
