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
  uint32_t magic, seq;
  double sim_t;
  float pos[3], vel[3], att[3];
  float bal[3];
  uint32_t hit, armed;
  uint16_t w, h;
};
struct CmdMsg {
  uint32_t magic, seq;
  float roll, pitch, yaw, thr;
  uint32_t armed, reset, seed;
};
#pragma pack(pop)
static const uint32_t MAGIC = 0x534D5241;
static const uint32_t MAGIC_CHASE = 0x45534843;
#pragma pack(push, 1)
struct ChaseHdr { uint32_t magic, seq; uint16_t idx, n, w, h; uint32_t off, len; };
#pragma pack(pop)

class ArmsBridge : public ModelPlugin {
 public:
  ~ArmsBridge() override { if (sock_ >= 0) close(sock_); }

  void Load(physics::ModelPtr model, sdf::ElementPtr sdf) override {
    model_ = model;
    world_ = model->GetWorld();
    auto get = [&](const char* k, double d) { return sdf->HasElement(k) ? sdf->Get<double>(k) : d; };
    camName_ = sdf->HasElement("camera") ? sdf->Get<std::string>("camera") : "cam";
    chaseName_ = sdf->HasElement("chase_camera") ? sdf->Get<std::string>("chase_camera") : "chase_cam";
    chaseModel_ = sdf->HasElement("chase_model") ? sdf->Get<std::string>("chase_model") : "chasecam";
    balloonName_ = sdf->HasElement("balloon") ? sdf->Get<std::string>("balloon") : "balloon";
    droneName_ = sdf->HasElement("drone") ? sdf->Get<std::string>("drone") : "drone_tgt";
    tagName_ = sdf->HasElement("tag") ? sdf->Get<std::string>("tag") : "tag";
    tip_ = {get("tip_x", 0.62), 0, get("tip_z", -0.06)};
    ropeTop_ = get("rope_top", -1.0);
    ropeBot_ = get("rope_bottom", -2.6);
    catchR_ = get("catch_r", 0.13);
    maxAngle_ = get("max_angle_deg", 35.0) * M_PI / 180;
    fullRate_ = get("full_stick_rate_dps", 400.0) * M_PI / 180;
    tauAtt_ = get("att_tau", 0.10);
    tauYaw_ = get("yaw_tau", 0.08);
    hoverThr_ = get("hover_thr", 0.5);
    drag_ = get("drag", 0.35);
    hitR_ = get("hit_radius", 1.2);
    int portIn = (int)get("port_in", 9202), portOut = (int)get("port_out", 9201);
    timeoutMs_ = (int)get("timeout_ms", 3000);

    sock_ = socket(AF_INET, SOCK_DGRAM, 0);
    sockaddr_in me{};
    me.sin_family = AF_INET;
    me.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    me.sin_port = htons(portIn);
    if (bind(sock_, (sockaddr*)&me, sizeof(me)) < 0) gzerr << "[arms] bind " << portIn << " 실패\n";
    peer_.sin_family = AF_INET;
    peer_.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    peer_.sin_port = htons(portOut);
    timeval tv{timeoutMs_ / 1000, (timeoutMs_ % 1000) * 1000};
    setsockopt(sock_, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));

    home_ = model_->WorldPose().Pos();
    NewEpisode(1);
    updConn_ = event::Events::ConnectWorldUpdateBegin(std::bind(&ArmsBridge::OnUpdate, this, std::placeholders::_1));
    gzmsg << "[arms] loaded in=" << portIn << " out=" << portOut << "\n";
  }

 private:
  void NewEpisode(uint32_t seed) {
    droneMode_ = (seed & 0x10000u) != 0;
    attachMode_ = (seed & 0x20000u) != 0;
    attached_ = false;
    seed &= 0xFFFFu;
    std::mt19937 rng(seed);
    std::uniform_real_distribution<double> U(0, 1);
    pos_ = home_; vel_ = {0, 0, 0};
    roll_ = pitch_ = 0; yaw_ = (U(rng) * 2 - 1) * M_PI; yawRate_ = 0;
    double brg = yaw_ + (U(rng) * 2 - 1) * 1.2;
    double dist = 18 + 10 * U(rng), alt = 6 + 6 * U(rng);
    if (attachMode_) {
      dist = 10 + 0.4 * (dist - 18);
      alt = 6 + (alt - 6) * 0.5;
    }
    if (droneMode_) {
      dist = 9 + 0.6 * (dist - 18);
      alt = 5 + (alt - 6) * 4.0 / 6.0;
    }
    bal0_ = {home_.X() + dist * std::cos(brg), home_.Y() + dist * std::sin(brg), alt};
    double wdir = U(rng) * 2 * M_PI, wspd = 0.3 + 1.0 * U(rng);
    wind_ = {wspd * std::cos(wdir), wspd * std::sin(wdir), 0};
    bobPh_ = U(rng) * 6.28;
    {
      auto dir = bal0_ - home_; dir.Z(0); dir.Normalize();
      side_ = {-dir.Y(), dir.X(), 0};
      camPos_ = {0, 0, -100};
    }
    double ddir = U(rng) * 2 * M_PI, dspd = 1.0 + 1.0 * U(rng);
    dvel_ = {dspd * std::cos(ddir), dspd * std::sin(ddir), 0};
    hit_ = false; armed_ = false;
    chaseYaw_ = yaw_;
    cmd_ = CmdMsg{};
    t0_ = world_->SimTime().Double();
  }

  ignition::math::Vector3d BalloonPos(double t) const {
    if (droneMode_) {
      auto side = ignition::math::Vector3d(-dvel_.Y(), dvel_.X(), 0).Normalized();
      auto p = bal0_ + dvel_ * t + side * (1.5 * std::sin(0.35 * t + bobPh_));
      p.Z(p.Z() + 0.4 * std::sin(0.5 * t + bobPh_));
      return p;
    }
    auto p = bal0_ + wind_ * t;
    p.Z(p.Z() + 0.6 * std::sin(0.4 * t + bobPh_));
    return p;
  }

  void TryAttachCamera() {
    if (!chaseConn_) {
      for (auto& s : sensors::SensorManager::Instance()->GetSensors()) {
        if (s->Name() != chaseName_) continue;
        auto cam = std::dynamic_pointer_cast<sensors::CameraSensor>(s);
        if (!cam || !cam->Camera()) break;
        chase_ = cam;
        chaseConn_ = cam->Camera()->ConnectNewImageFrame(
            [this](const unsigned char* img, unsigned w, unsigned h, unsigned, const std::string&) {
              std::lock_guard<std::mutex> lk(mtx_);
              chaseFrame_.assign(img, img + w * h * 3);
              cw_ = w; ch_ = h;
            });
      }
    }
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
      gzmsg << "[arms] camera attached\n";
    }
  }

  static double Wrap(double a) { return std::atan2(std::sin(a), std::cos(a)); }

  void OnUpdate(const common::UpdateInfo& info) {
    TryAttachCamera();
    double now = info.simTime.Double();
    double dt = lastT_ < 0 ? 0 : now - lastT_;
    lastT_ = now;
    double t = now - t0_;

    auto bp = BalloonPos(t);
    std::vector<uint8_t> img;
    unsigned w = 0, h = 0;
    {
      std::lock_guard<std::mutex> lk(mtx_);
      if (ready_) { img.swap(frame_); w = fw_; h = fh_; ready_ = false; }
    }
    if (!img.empty()) Exchange(img, w, h, t, bp);
    t = now - t0_;
    bp = BalloonPos(t);

    if (dt > 0) Step(dt);

    if (!hit_ && (bp - pos_).Length() < hitR_) {
      hit_ = true;
      gzmsg << "[arms] HIT t=" << t << "\n";
    }
    ignition::math::Quaterniond qd(roll_, pitch_, yaw_);
    auto tip = pos_ + qd.RotateVector(tip_);
    if (attachMode_ && !attached_) {
      double dz = tip.Z() - bp.Z();
      double dh = std::hypot(tip.X() - bp.X(), tip.Y() - bp.Y());
      if (dz < ropeTop_ - 0.05 && dz > ropeBot_ + 0.05 && dh < catchR_) {
        attached_ = true;
        tagOff_ = dz;
        tagYaw_ = yaw_;
        gzmsg << "[arms] TAG ATTACHED t=" << t << " dh=" << dh << "\n";
      }
    }
    const ignition::math::Vector3d away(0, 0, -50);
    double hdg = std::atan2(dvel_.Y(), dvel_.X());
    if (auto b = world_->ModelByName(balloonName_))
      b->SetWorldPose({((hit_ && !attachMode_) || droneMode_) ? away : bp, {0, 0, 0}});
    if (auto g = world_->ModelByName(tagName_)) {
      if (!attachMode_) g->SetWorldPose({away + ignition::math::Vector3d(0, 0, -10), {0, 0, 0}});
      else if (attached_) g->SetWorldPose({bp + ignition::math::Vector3d(0, 0, tagOff_), {0, 0, tagYaw_}});
      else g->SetWorldPose({tip, {roll_, pitch_, yaw_}});
    }
    if (auto d = world_->ModelByName(droneName_))
      d->SetWorldPose({(hit_ || !droneMode_) ? away : bp, {0, 0, hdg}});
    model_->SetWorldPose({pos_, {roll_, pitch_, yaw_}});
    auto cm = world_->ModelByName(chaseModel_);
    if (cm && attachMode_) {
      auto rope = bp + ignition::math::Vector3d(0, 0, 0.5 * (ropeTop_ + ropeBot_));
      auto look = attached_ ? rope * 0.6 + pos_ * 0.4 : rope * 0.5 + pos_ * 0.5;
      double span = std::min(8.0, std::max(3.2, 0.6 * (rope - pos_).Length() + 2.2));
      auto want = look + side_ * span + ignition::math::Vector3d(0, 0, 0.6);
      if (camPos_.Z() < -50) camPos_ = want;
      camPos_ += (want - camPos_) * std::min(1.0, dt / 0.7);
      auto d = look - camPos_;
      cm->SetWorldPose({camPos_, {0, std::atan2(-d.Z(), std::hypot(d.X(), d.Y())), std::atan2(d.Y(), d.X())}});
    } else if (auto c = cm) {
      if (dt > 0) chaseYaw_ += Wrap(yaw_ - chaseYaw_) * std::min(1.0, dt / 0.8);
      ignition::math::Vector3d cp = pos_ + ignition::math::Vector3d(-6 * std::cos(chaseYaw_), -6 * std::sin(chaseYaw_), 2.5);
      if (cp.Z() < 0.5) cp.Z(0.5);
      auto d = pos_ + ignition::math::Vector3d(0, 0, 1.0) - cp;
      c->SetWorldPose({cp, {0, std::atan2(-d.Z(), std::hypot(d.X(), d.Y())), std::atan2(d.Y(), d.X())}});
    }
  }

  void Step(double dt) {
    const double g = 9.81;
    double rs = 0, ps = 0, yr = 0, thr = 0;
    if (armed_) {
      rs = std::clamp((double)cmd_.roll, -1.0, 1.0) * maxAngle_;
      ps = std::clamp((double)cmd_.pitch, -1.0, 1.0) * maxAngle_;
      yr = -std::clamp((double)cmd_.yaw, -1.0, 1.0) * fullRate_;
      thr = std::clamp((double)cmd_.thr, 0.0, 1.0);
    }
    double a = std::min(1.0, dt / tauAtt_);
    roll_ += (rs - roll_) * a;
    pitch_ += (ps - pitch_) * a;
    yawRate_ += (yr - yawRate_) * std::min(1.0, dt / tauYaw_);
    yaw_ = Wrap(yaw_ + yawRate_ * dt);

    ignition::math::Quaterniond q(roll_, pitch_, yaw_);
    auto zb = q.RotateVector({0, 0, 1});
    double T = thr / hoverThr_ * g;
    ignition::math::Vector3d acc = zb * T - ignition::math::Vector3d(0, 0, g) - vel_ * drag_;
    vel_ += acc * dt;
    pos_ += vel_ * dt;
    if (pos_.Z() < home_.Z()) {
      pos_.Z(home_.Z());
      if (vel_.Z() < 0) vel_ = {0, 0, 0};
      if (!armed_) { roll_ = pitch_ = 0; yawRate_ = 0; }
    }
  }

  void Exchange(const std::vector<uint8_t>& img, unsigned w, unsigned h, double t,
                const ignition::math::Vector3d& bp) {
    FrameHdr hd{};
    hd.magic = MAGIC; hd.seq = ++seq_; hd.sim_t = t;
    float p[3] = {(float)pos_.X(), (float)pos_.Y(), (float)pos_.Z()};
    float v[3] = {(float)vel_.X(), (float)vel_.Y(), (float)vel_.Z()};
    float at[3] = {(float)roll_, (float)pitch_, (float)yaw_};
    float b[3] = {(float)bp.X(), (float)bp.Y(), (float)bp.Z()};
    std::memcpy(hd.pos, p, 12); std::memcpy(hd.vel, v, 12);
    std::memcpy(hd.att, at, 12); std::memcpy(hd.bal, b, 12);
    hd.hit = (hit_ ? 1u : 0u) | (attached_ ? 2u : 0u); hd.armed = armed_; hd.w = w; hd.h = h;
    SendChase();
    std::vector<uint8_t> pkt(sizeof(hd) + img.size());
    std::memcpy(pkt.data(), &hd, sizeof(hd));
    std::memcpy(pkt.data() + sizeof(hd), img.data(), img.size());
    sendto(sock_, pkt.data(), pkt.size(), 0, (sockaddr*)&peer_, sizeof(peer_));

    CmdMsg m{};
    while (true) {
      ssize_t n = recv(sock_, &m, sizeof(m), 0);
      if (n < 0) { cmd_ = CmdMsg{}; armed_ = false; return; }
      if (n == sizeof(m) && m.magic == MAGIC && m.seq == seq_) break;
    }
    if (m.reset) { NewEpisode(m.seed); return; }
    cmd_ = m;
    armed_ = m.armed;
  }

  void SendChase() {
    std::vector<uint8_t> f;
    unsigned w, h;
    {
      std::lock_guard<std::mutex> lk(mtx_);
      if (chaseFrame_.empty()) return;
      f = chaseFrame_; w = cw_; h = ch_;
    }
    const uint32_t CH = 60000;
    uint16_t n = (f.size() + CH - 1) / CH;
    for (uint16_t i = 0; i < n; ++i) {
      uint32_t off = i * CH, len = std::min<uint32_t>(CH, f.size() - off);
      ChaseHdr c{MAGIC_CHASE, seq_, i, n, (uint16_t)w, (uint16_t)h, off, len};
      std::vector<uint8_t> pkt(sizeof(c) + len);
      std::memcpy(pkt.data(), &c, sizeof(c));
      std::memcpy(pkt.data() + sizeof(c), f.data() + off, len);
      sendto(sock_, pkt.data(), pkt.size(), 0, (sockaddr*)&peer_, sizeof(peer_));
    }
  }

  physics::ModelPtr model_;
  physics::WorldPtr world_;
  sensors::CameraSensorPtr cam_;
  event::ConnectionPtr updConn_, camConn_, chaseConn_;
  sensors::CameraSensorPtr chase_;
  std::string chaseName_, chaseModel_;
  std::vector<uint8_t> chaseFrame_;
  unsigned cw_ = 0, ch_ = 0;
  double chaseYaw_ = 0;
  std::string camName_, balloonName_, droneName_;
  bool droneMode_ = false, attachMode_ = false, attached_ = false;
  std::string tagName_;
  ignition::math::Vector3d tip_, side_, camPos_;
  double ropeTop_ = -1.0, ropeBot_ = -2.6, catchR_ = 0.13, tagOff_ = -1.8, tagYaw_ = 0;
  int sock_ = -1, timeoutMs_ = 3000;
  sockaddr_in peer_{};
  std::mutex mtx_;
  std::vector<uint8_t> frame_;
  unsigned fw_ = 0, fh_ = 0;
  bool ready_ = false, hit_ = false, armed_ = false;
  uint32_t seq_ = 0;
  CmdMsg cmd_{};
  double maxAngle_, fullRate_, tauAtt_, tauYaw_, hoverThr_, drag_, hitR_;
  double roll_ = 0, pitch_ = 0, yaw_ = 0, yawRate_ = 0, t0_ = 0, lastT_ = -1, bobPh_ = 0;
  ignition::math::Vector3d home_, pos_, vel_, bal0_, wind_, dvel_;
};

GZ_REGISTER_MODEL_PLUGIN(ArmsBridge)
}
