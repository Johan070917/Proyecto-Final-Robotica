/*
 * ============================================================
 *  esp32_motors.ino  -  ESCLAVO de motores del robot diferencial
 *  H-Bridge L298N + 2 motores 12V con encoders Hall en cuadratura
 *
 *  Maestro: Raspberry Pi 5 (ROS 2).
 *  Comunicacion: USB CDC (Serial), 115200 baud, lineas terminadas en '\n'.
 * ============================================================
 *
 *  PINES (mantienen el pinout de motores_mov.txt):
 *
 *  MOTOR A (Izquierdo)
 *    IN1  -> GPIO 25
 *    IN2  -> GPIO 26
 *    ENA  -> GPIO 27  (PWM)
 *
 *  MOTOR B (Derecho)
 *    IN3  -> GPIO 14
 *    IN4  -> GPIO 12
 *    ENB  -> GPIO 13  (PWM)
 *
 *  ENCODER MOTOR A
 *    S1   -> GPIO 34  (interrupt, canal A)
 *    S2   -> GPIO 35  (canal B, direccion)
 *
 *  ENCODER MOTOR B
 *    S1   -> GPIO 32  (interrupt, canal A)
 *    S2   -> GPIO 33  (canal B, direccion)
 *
 * ============================================================
 *  PROTOCOLO
 *
 *  RPi -> ESP32:
 *    PWM <left> <right>     left,right en [-255, 255]
 *    STOP                   detiene motores y resetea PWM a 0
 *    RESET                  resetea contadores de encoder a 0
 *    PING                   responde PONG
 *
 *  ESP32 -> RPi (automatico cada 20 ms):
 *    ENC <left> <right> <dt_us>
 *
 *  ESP32 -> RPi (en respuesta):
 *    READY                  al arrancar
 *    OK                     comando aceptado
 *    PONG                   respuesta a PING
 *    ERR:<causa>            error de parseo
 *
 *  WATCHDOG:
 *    Si la ESP32 no recibe ningun comando PWM en 500 ms,
 *    detiene los motores automaticamente.
 * ============================================================
 */

// ---- PINES MOTOR A (Izquierdo) ----
#define MOTOR_A_IN1     25
#define MOTOR_A_IN2     26
#define MOTOR_A_ENA     27

// ---- PINES MOTOR B (Derecho) ----
#define MOTOR_B_IN3     14
#define MOTOR_B_IN4     12
#define MOTOR_B_ENB     13

// ---- PINES ENCODER A ----
#define ENC_A_S1        34
#define ENC_A_S2        35

// ---- PINES ENCODER B ----
#define ENC_B_S1        32
#define ENC_B_S2        33

// ---- PWM ----
#define PWM_FREQ        1000
#define PWM_RESOLUTION  8       // 0..255

// ---- WATCHDOG ----
const unsigned long WATCHDOG_MS = 500;

// ---- TELEMETRIA ----
const unsigned long ENC_REPORT_US = 20000;   // 20 ms = 50 Hz

// ---- ESTADO ----
volatile long encA_count = 0;
volatile long encB_count = 0;
unsigned long last_cmd_ms = 0;
unsigned long last_enc_us = 0;
long          last_encA   = 0;
long          last_encB   = 0;

// ============================================================
//  ISR encoders
// ============================================================
void IRAM_ATTR ISR_EncoderA() {
  if (digitalRead(ENC_A_S2) == HIGH) encA_count++;
  else                                encA_count--;
}

void IRAM_ATTR ISR_EncoderB() {
  if (digitalRead(ENC_B_S2) == HIGH) encB_count++;
  else                                encB_count--;
}

// ============================================================
//  Motores
// ============================================================
void setMotorA(int spd) {
  if (spd > 255)  spd = 255;
  if (spd < -255) spd = -255;
  if (spd > 0) {
    digitalWrite(MOTOR_A_IN1, HIGH);
    digitalWrite(MOTOR_A_IN2, LOW);
    ledcWrite(MOTOR_A_ENA, spd);
  } else if (spd < 0) {
    digitalWrite(MOTOR_A_IN1, LOW);
    digitalWrite(MOTOR_A_IN2, HIGH);
    ledcWrite(MOTOR_A_ENA, -spd);
  } else {
    digitalWrite(MOTOR_A_IN1, LOW);
    digitalWrite(MOTOR_A_IN2, LOW);
    ledcWrite(MOTOR_A_ENA, 0);
  }
}

void setMotorB(int spd) {
  if (spd > 255)  spd = 255;
  if (spd < -255) spd = -255;
  if (spd > 0) {
    digitalWrite(MOTOR_B_IN3, HIGH);
    digitalWrite(MOTOR_B_IN4, LOW);
    ledcWrite(MOTOR_B_ENB, spd);
  } else if (spd < 0) {
    digitalWrite(MOTOR_B_IN3, LOW);
    digitalWrite(MOTOR_B_IN4, HIGH);
    ledcWrite(MOTOR_B_ENB, -spd);
  } else {
    digitalWrite(MOTOR_B_IN3, LOW);
    digitalWrite(MOTOR_B_IN4, LOW);
    ledcWrite(MOTOR_B_ENB, 0);
  }
}

void stopAll() {
  setMotorA(0);
  setMotorB(0);
}

// ============================================================
//  Parser
// ============================================================
void sendLine(const String& s) {
  Serial.print(s);
  Serial.print('\n');
}

void handleLine(String line) {
  line.trim();
  if (line.length() == 0) return;

  if (line.startsWith("PWM ")) {
    int sp1 = line.indexOf(' ', 4);
    if (sp1 < 0) { sendLine("ERR:bad_args"); return; }
    int left  = line.substring(4, sp1).toInt();
    int right = line.substring(sp1 + 1).toInt();
    setMotorA(left);
    setMotorB(right);
    last_cmd_ms = millis();
    sendLine("OK");
  }
  else if (line == "STOP") {
    stopAll();
    last_cmd_ms = millis();
    sendLine("OK");
  }
  else if (line == "RESET") {
    noInterrupts();
    encA_count = 0;
    encB_count = 0;
    last_encA  = 0;
    last_encB  = 0;
    interrupts();
    sendLine("OK");
  }
  else if (line == "PING") {
    sendLine("PONG");
  }
  else if (line == "WHOAMI") {
    sendLine("ID:MOTORS");
  }
  else {
    sendLine("ERR:unknown_cmd");
  }
}

// ============================================================
//  Setup
// ============================================================
void setup() {
  pinMode(MOTOR_A_IN1, OUTPUT);
  pinMode(MOTOR_A_IN2, OUTPUT);
  pinMode(MOTOR_B_IN3, OUTPUT);
  pinMode(MOTOR_B_IN4, OUTPUT);

  ledcAttach(MOTOR_A_ENA, PWM_FREQ, PWM_RESOLUTION);
  ledcAttach(MOTOR_B_ENB, PWM_FREQ, PWM_RESOLUTION);

  // GPIO 34 y 35 son input-only y NO tienen pull-up interno en el ESP32.
  // Si se usan, deben llevar un resistor externo de 10 kohm a 3.3V.
  // Por eso aqui usamos INPUT (sin pull-up) — el pull-up es externo.
  pinMode(ENC_A_S1, INPUT);
  pinMode(ENC_A_S2, INPUT);
  // GPIO 32 y 33 si soportan pull-up interno.
  pinMode(ENC_B_S1, INPUT_PULLUP);
  pinMode(ENC_B_S2, INPUT_PULLUP);

  attachInterrupt(digitalPinToInterrupt(ENC_A_S1), ISR_EncoderA, RISING);
  attachInterrupt(digitalPinToInterrupt(ENC_B_S1), ISR_EncoderB, RISING);

  stopAll();
  Serial.begin(115200);
  delay(200);
  last_cmd_ms = millis();
  last_enc_us = micros();
  sendLine("READY");
}

// ============================================================
//  Loop
// ============================================================
static String rx;

void loop() {
  // --- Recepcion no bloqueante ---
  while (Serial.available()) {
    char c = (char)Serial.read();
    if (c == '\n' || c == '\r') {
      if (rx.length() > 0) { handleLine(rx); rx = ""; }
    } else {
      rx += c;
      if (rx.length() > 64) rx = "";
    }
  }

  // --- Watchdog ---
  if (millis() - last_cmd_ms > WATCHDOG_MS) {
    stopAll();
  }

  // --- Telemetria de encoders cada 20 ms ---
  unsigned long now_us = micros();
  if (now_us - last_enc_us >= ENC_REPORT_US) {
    unsigned long dt = now_us - last_enc_us;
    last_enc_us = now_us;

    noInterrupts();
    long a = encA_count;
    long b = encB_count;
    interrupts();

    // Reportamos posiciones absolutas (la RPi calcula deltas).
    Serial.print("ENC ");
    Serial.print(a);
    Serial.print(' ');
    Serial.print(b);
    Serial.print(' ');
    Serial.println(dt);
  }
}
