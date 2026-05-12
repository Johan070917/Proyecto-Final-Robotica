/*
 * ============================================================
 *  esp32_lift.ino  -  ESCLAVO del montacargas
 *
 *  Subsistemas:
 *    - Motoreductor 12V 25GA370 (60 RPM) con encoder Hall, controlado
 *      por driver TB6612FNG.  Sube y baja el montacargas.
 *    - Servo MG995 que rota la pala / aspa entre vertical (recogida)
 *      y horizontal (extendida, paralela al suelo).
 *
 *  Maestro: Raspberry Pi 5 (ROS 2).
 *  Comunicacion: USB CDC (Serial), 115200 baud.
 * ============================================================
 *
 *  PINES (ESP32):
 *
 *  TB6612FNG (motor del lift)
 *    AIN1   -> GPIO 25
 *    AIN2   -> GPIO 26
 *    PWMA   -> GPIO 27
 *    STBY   -> GPIO 33   (HIGH = activo)
 *
 *  Encoder Hall del 25GA370
 *    A      -> GPIO 34   (interrupt)
 *    B      -> GPIO 35   (direccion)
 *
 *  Servo MG995
 *    SIGNAL -> GPIO 13
 *
 * ============================================================
 *  PROTOCOLO
 *
 *  RPi -> ESP32:
 *    LIFT_UP    <ticks>     sube hasta acumular <ticks> de encoder
 *    LIFT_DOWN  <ticks>     baja hasta acumular <ticks> de encoder
 *    LIFT_STOP              detiene el motor
 *    LIFT_HOME              baja a posicion 0 (referencia inicial)
 *    FORKS_OPEN             servo a posicion HORIZONTAL (paralela al suelo)
 *    FORKS_CLOSE            servo a posicion VERTICAL (recogida)
 *    ANGLE <deg>            mueve el servo a un angulo especifico (calibracion)
 *    STATUS?                pide el estado actual
 *    PING                   responde PONG
 *
 *  ESP32 -> RPi:
 *    READY                  al arrancar
 *    OK                     comando aceptado
 *    DONE                   operacion finalizada
 *    ERR:<causa>            error
 *    STATUS:pos=<n>,run=<0|1>
 *    PONG                   respuesta a PING
 *
 *  Nota: la posicion "pos" es un contador relativo de ticks.
 *  Vale 0 al encender la ESP32; el robot debe arrancar siempre con
 *  el montacargas en su posicion mas baja.
 * ============================================================
 */

#include <ESP32Servo.h>

// ---- TB6612FNG ----
#define TB_AIN1     25
#define TB_AIN2     26
#define TB_PWMA     27
#define TB_STBY     33

// ---- Encoder Hall del 25GA370 ----
#define ENC_LIFT_A  34
#define ENC_LIFT_B  35

// ---- Servo MG995 ----
#define PIN_SERVO   13

// ---- PWM ----
#define PWM_FREQ        20000  // 20 kHz para que no se oiga
#define PWM_RESOLUTION  8

// ---- Velocidad del lift ----
// Subimos a 250/220 para asegurar torque suficiente.  Si el motor se
// queja o la mecanica se fuerza, baja estos valores.
const int PWM_LIFT_UP   = 250;   // 0..255  (subir contra gravedad)
const int PWM_LIFT_DOWN = 220;   // 0..255  (bajar, ayuda la gravedad)
const int PWM_HOMING    = 120;

// ---- Servo (calibrar al instalar las palas) ----
const int SERVO_VERTICAL    = 80;    // pala perpendicular al suelo (recogida)
const int SERVO_HORIZONTAL  = 175;   // pala paralela al suelo (extendida)

// ---- Duracion fija del movimiento del lift (modo sin encoder) ----
const unsigned long LIFT_DURATION_MS = 2000;

// ============================================================
//  Estado
// ============================================================
Servo servoPalas;

volatile long enc_count = 0;
long  target_ticks  = 0;     // ticks objetivo de la operacion en curso
int   lift_dir      = 0;     // +1 sube, -1 baja, 0 parado
bool  lift_running  = false;
unsigned long t_op_start = 0;

// ============================================================
//  ISR del encoder
// ============================================================
void IRAM_ATTR ISR_EncoderLift() {
  // Suponemos que cuando el motor sube, B es HIGH al subir flanco de A.
  // Si invertido en tu mecanica, cambia los signos aqui.
  if (digitalRead(ENC_LIFT_B) == HIGH) enc_count++;
  else                                  enc_count--;
}

// ============================================================
//  Driver TB6612FNG
// ============================================================
void liftDrive(int dir, int pwm) {
  // dir: +1 sube, -1 baja, 0 freno
  // Polaridad invertida respecto al cableado original: con esta logica
  // LIFT_UP fisicamente sube y LIFT_DOWN fisicamente baja.
  digitalWrite(TB_STBY, HIGH);
  if (dir > 0) {
    digitalWrite(TB_AIN1, LOW);
    digitalWrite(TB_AIN2, HIGH);
    ledcWrite(TB_PWMA, pwm);
  } else if (dir < 0) {
    digitalWrite(TB_AIN1, HIGH);
    digitalWrite(TB_AIN2, LOW);
    ledcWrite(TB_PWMA, pwm);
  } else {
    // Freno corto
    digitalWrite(TB_AIN1, HIGH);
    digitalWrite(TB_AIN2, HIGH);
    ledcWrite(TB_PWMA, 0);
  }
}

void liftStop() {
  liftDrive(0, 0);
  lift_running = false;
  lift_dir = 0;
  target_ticks = 0;
}

// ============================================================
//  Comandos
// ============================================================
void sendLine(const String& s) {
  Serial.print(s);
  Serial.print('\n');
}

long parseLongAfter(const String& line, int from_idx) {
  String num = line.substring(from_idx);
  num.trim();
  return num.toInt();
}

void cmdLiftMove(int dir, long ticks) {
  // Modo sin encoder: el parametro <ticks> se ignora.
  // El motor se mueve durante LIFT_DURATION_MS y se detiene solo.
  (void)ticks;
  lift_dir = dir;
  lift_running = true;
  t_op_start = millis();
  liftDrive(dir, dir > 0 ? PWM_LIFT_UP : PWM_LIFT_DOWN);
  sendLine("OK");
}

void cmdLiftHome() {
  // Sin encoder no hay referencia absoluta: tratamos HOME como "baja 2s".
  lift_dir = -1;
  lift_running = true;
  t_op_start = millis();
  liftDrive(-1, PWM_HOMING);
  sendLine("OK");
}

void cmdForksOpen() {
  servoPalas.write(SERVO_HORIZONTAL);
  delay(450);
  sendLine("DONE");
}

void cmdForksClose() {
  servoPalas.write(SERVO_VERTICAL);
  delay(450);
  sendLine("DONE");
}

void handleLine(String line) {
  line.trim();
  if (line.length() == 0) return;

  if (line.startsWith("LIFT_UP")) {
    long n = parseLongAfter(line, 7);
    if (n <= 0) { sendLine("ERR:bad_ticks"); return; }
    cmdLiftMove(+1, n);
  }
  else if (line.startsWith("LIFT_DOWN")) {
    long n = parseLongAfter(line, 9);
    if (n <= 0) { sendLine("ERR:bad_ticks"); return; }
    cmdLiftMove(-1, n);
  }
  else if (line == "LIFT_STOP") {
    liftStop();
    sendLine("OK");
    sendLine("DONE");
  }
  else if (line == "LIFT_HOME") {
    cmdLiftHome();
  }
  else if (line == "FORKS_OPEN") {
    sendLine("OK");
    cmdForksOpen();
  }
  else if (line == "FORKS_CLOSE") {
    sendLine("OK");
    cmdForksClose();
  }
  else if (line.startsWith("ANGLE ")) {
    int deg = line.substring(6).toInt();
    servoPalas.write(deg);
    sendLine("DONE");
  }
  else if (line == "STATUS?") {
    noInterrupts();
    long pos = enc_count;
    interrupts();
    String s = "STATUS:pos=" + String(pos) +
               ",run="      + String(lift_running ? 1 : 0);
    sendLine(s);
  }
  else if (line == "PING") {
    sendLine("PONG");
  }
  else if (line == "WHOAMI") {
    sendLine("ID:LIFT");
  }
  else {
    sendLine("ERR:unknown_cmd");
  }
}

// ============================================================
//  Setup
// ============================================================
void setup() {
  // ---- Anclar el pin del servo a LOW lo ANTES posible ----
  // Si la bateria del servo se enciende mientras GPIO 13 esta flotando,
  // el MG995 ve pulsos aleatorios y se sacude descontroladamente.
  // Forzando LOW, el servo no recibe pulsos validos y se queda quieto
  // hasta que ESP32Servo tome el pin.
  pinMode(PIN_SERVO, OUTPUT);
  digitalWrite(PIN_SERVO, LOW);

  // ---- Reservar timers para ESP32Servo ANTES del attach ----
  // Sin esto, attach() falla silenciosamente y la senal PWM no se genera
  // (porque ledcAttach del TB6612FNG ya consume timer 0).
  ESP32PWM::allocateTimer(0);
  ESP32PWM::allocateTimer(1);
  ESP32PWM::allocateTimer(2);
  ESP32PWM::allocateTimer(3);

  // ---- Servo (inicializar primero, antes de ledc del motor) ----
  servoPalas.setPeriodHertz(50);
  servoPalas.attach(PIN_SERVO, 500, 2400);
  servoPalas.write(SERVO_VERTICAL);   // arranca recogido

  // ---- TB6612FNG ----
  pinMode(TB_AIN1, OUTPUT);
  pinMode(TB_AIN2, OUTPUT);
  pinMode(TB_STBY, OUTPUT);
  digitalWrite(TB_STBY, LOW);     // STBY=LOW al inicio (driver dormido)

  ledcAttach(TB_PWMA, PWM_FREQ, PWM_RESOLUTION);
  ledcWrite(TB_PWMA, 0);

  // ---- Encoder (DESACTIVADO: control por tiempo, no por ticks) ----
  // Se deja el codigo de la ISR arriba por si en el futuro se reactiva.
  // pinMode(ENC_LIFT_A, INPUT_PULLUP);
  // pinMode(ENC_LIFT_B, INPUT_PULLUP);
  // attachInterrupt(digitalPinToInterrupt(ENC_LIFT_A), ISR_EncoderLift, RISING);

  Serial.begin(115200);
  delay(200);
  sendLine("READY");
}

// ============================================================
//  Loop
// ============================================================
static String rx;

void loop() {
  // ---- Recepcion ----
  while (Serial.available()) {
    char c = (char)Serial.read();
    if (c == '\n' || c == '\r') {
      if (rx.length() > 0) { handleLine(rx); rx = ""; }
    } else {
      rx += c;
      if (rx.length() > 64) rx = "";
    }
  }

  // ---- Control del lift (sin encoder, por tiempo) ----
  if (lift_running) {
    if (millis() - t_op_start >= LIFT_DURATION_MS) {
      liftStop();
      sendLine("DONE");
    }
  }
}
