#include <Wire.h>
#include <WiFi.h>
#include <WiFiUdp.h>

const char* SSID = "YOUR_HOTSPOT";
const char* PASS = "YOUR_PASSWORD";
const uint16_t PORT = 9400;
#define SDA_PIN 21
#define SCL_PIN 22
const uint8_t MPU = 0x68;
const float ALPHA = 0.98f;

WiFiUDP udp;
bool wifiOk = false;
float roll = 0, pitch = 0, yaw = 0;
float gbx = 0, gby = 0, gbz = 0;
uint32_t lastUs;

void mpuWrite(uint8_t reg, uint8_t v) {
  Wire.beginTransmission(MPU); Wire.write(reg); Wire.write(v); Wire.endTransmission();
}

bool mpuRead(int16_t* a, int16_t* g) {
  Wire.beginTransmission(MPU); Wire.write(0x3B);
  if (Wire.endTransmission(false) != 0) return false;
  if (Wire.requestFrom(MPU, (uint8_t)14) != 14) return false;
  int16_t v[7];
  for (int i = 0; i < 7; i++) v[i] = (Wire.read() << 8) | Wire.read();
  a[0] = v[0]; a[1] = v[1]; a[2] = v[2];
  g[0] = v[4]; g[1] = v[5]; g[2] = v[6];
  return true;
}

void setup() {
  Serial.begin(115200);
  Wire.begin(SDA_PIN, SCL_PIN, 400000);
  mpuWrite(0x6B, 0x00);
  mpuWrite(0x1A, 0x03);
  mpuWrite(0x1B, 0x08);
  mpuWrite(0x1C, 0x00);
  delay(100);

  int16_t a[3], g[3]; int n = 0;
  for (int i = 0; i < 200; i++) {
    if (mpuRead(a, g)) { gbx += g[0]; gby += g[1]; gbz += g[2]; n++; }
    delay(5);
  }
  if (n) { gbx /= n; gby /= n; gbz /= n; }
  Serial.printf("# gyro bias %.1f %.1f %.1f (n=%d)\n", gbx, gby, gbz, n);

  WiFi.mode(WIFI_STA);
  WiFi.begin(SSID, PASS);
  for (int i = 0; i < 40 && WiFi.status() != WL_CONNECTED; i++) delay(250);
  wifiOk = WiFi.status() == WL_CONNECTED;
  Serial.println(wifiOk ? "# wifi ok " + WiFi.localIP().toString() : "# wifi FAIL -> serial only");
  if (wifiOk) udp.begin(PORT);
  lastUs = micros();
}

void loop() {
  int16_t a[3], g[3];
  if (!mpuRead(a, g)) { delay(2); return; }
  uint32_t now = micros();
  float dt = (now - lastUs) * 1e-6f; lastUs = now;
  float gx = (g[0] - gbx) / 65.5f, gy = (g[1] - gby) / 65.5f, gz = (g[2] - gbz) / 65.5f;
  float accRoll = atan2f(a[1], a[2]) * 57.2958f;
  float accPitch = atan2f(-a[0], sqrtf((float)a[1] * a[1] + (float)a[2] * a[2])) * 57.2958f;
  roll = ALPHA * (roll + gx * dt) + (1 - ALPHA) * accRoll;
  pitch = ALPHA * (pitch + gy * dt) + (1 - ALPHA) * accPitch;
  yaw += gz * dt;

  static uint32_t lastSend = 0;
  if (now - lastSend >= 10000) {
    lastSend = now;
    char line[48];
    int len = snprintf(line, sizeof(line), "%.2f,%.2f,%.2f\n", roll, pitch, yaw);
    Serial.write(line, len);
    if (wifiOk) {
      udp.beginPacket(IPAddress(255, 255, 255, 255), PORT);
      udp.write((const uint8_t*)line, len);
      udp.endPacket();
    }
  }
}
