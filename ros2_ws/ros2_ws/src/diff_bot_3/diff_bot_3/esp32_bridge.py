"""
esp32_bridge.py  -  Puente serie generico RPi5 (maestro) <-> ESP32 (esclavo).

Se lanza una instancia por cada ESP32:
  * esp32_bridge_motors  -> /motors_cmd, /motors_status   (expected_id=MOTORS)
  * esp32_bridge_lift    -> /lift_cmd,   /lift_status     (expected_id=LIFT)

Parametros:
  port          puerto serie (string)
                  - "auto" (por defecto): escanea /dev/ttyUSB* y /dev/ttyACM*
                    y se queda con el que responda WHOAMI -> ID:<expected_id>
                  - ruta explicita (ej: "/dev/ttyUSB3"): la usa tal cual
  expected_id   identificador esperado al enviar WHOAMI ("MOTORS" o "LIFT").
                Solo se usa si port="auto".
  baud          baudios (int, default 115200)
  cmd_topic     topic de entrada (string)   -> reescribe a serial
  status_topic  topic de salida (string)    <- lee de serial

El protocolo es texto plano por lineas (terminadas en '\n').

Cada firmware ESP32 debe responder a WHOAMI:
  motors firmware -> "ID:MOTORS"
  lift firmware   -> "ID:LIFT"

Diseno:
  - Las suscripciones y publicaciones se crean INMEDIATAMENTE en __init__,
    antes de buscar el puerto serie. Asi el resto del sistema siempre ve
    al bridge como subscriber/publisher valido, aunque la ESP32 todavia
    no este lista. Mientras self.ser sea None, los comandos entrantes se
    descartan con un warning periodico.
  - La autodeteccion del puerto corre en un hilo aparte, con reintentos
    (hasta AUTODETECT_BUDGET_S segundos en total).
  - Si un puerto esta ocupado momentaneamente (otro bridge probandolo,
    ModemManager), se reintenta despues de un breve backoff.
  - Una vez identificado el puerto, se MANTIENE abierto el Serial
    (no se cierra y reabre), evitando un segundo reset de 2s del CDC
    y posibles condiciones de carrera con otros bridges.
"""

import glob
import threading
import time

import rclpy
from rclpy.node import Node
from std_msgs.msg import String

try:
    import serial
except ImportError:
    serial = None


# ---------------- Config de la autodeteccion ----------------
AUTODETECT_BUDGET_S = 60.0   # tiempo total maximo escaneando
PROBE_OPEN_TIMEOUT  = 0.3    # timeout de lectura al probar
PROBE_CDC_RESET_S   = 2.0    # tiempo que tarda la ESP32 en quedar lista
PROBE_REPLY_WAIT_S  = 1.5    # tiempo maximo esperando respuesta a WHOAMI
RETRY_BUSY_DELAY_S  = 0.7    # cuanto esperar antes de reintentar un puerto ocupado


def _is_busy_error(exc: Exception) -> bool:
    msg = str(exc).lower()
    return ('busy' in msg) or ('errno 16' in msg) or ('resource' in msg)


def _probe_port(port: str, baud: int, expected_id: str, logger=None):
    """Intenta abrir el puerto y verificar que responde ID:<expected_id>.

    Devuelve:
      * objeto serial.Serial abierto -> match correcto, dejar este Serial
        en uso (NO cerrarlo)
      * False -> abrio bien pero el ID no coincide (o no respondio nada)
      * None  -> no se pudo abrir (busy / no existe / permiso)
    """
    target = f"ID:{expected_id}".strip().upper()
    s = None
    try:
        s = serial.Serial(port, baud, timeout=PROBE_OPEN_TIMEOUT)
    except Exception as e:
        if logger is not None and not _is_busy_error(e):
            logger.warn(f"probe {port}: {e}")
        return None

    try:
        time.sleep(PROBE_CDC_RESET_S)
        s.reset_input_buffer()
        s.write(b"WHOAMI\n")
        s.flush()

        t0 = time.time()
        while time.time() - t0 < PROBE_REPLY_WAIT_S:
            raw = s.readline()
            if not raw:
                continue
            line = raw.decode('ascii', errors='ignore').strip().upper()
            if not line:
                continue
            if line == target:
                return s
            # Ignoramos READY, PONG, ERR, ID:<otro>, etc.
        try:
            s.close()
        except Exception:
            pass
        return False
    except Exception as e:
        if logger is not None:
            logger.warn(f"probe {port}: {e}")
        try:
            s.close()
        except Exception:
            pass
        return False


def _autodetect_port(expected_id: str, baud: int, logger=None,
                     should_continue=lambda: True):
    """Escanea puertos hasta encontrar la ESP32 cuyo WHOAMI coincida.

    Devuelve (path, serial.Serial) o ('', None) si se acaba el tiempo
    o should_continue() devuelve False.
    """
    deadline = time.time() + AUTODETECT_BUDGET_S
    announced = False

    while time.time() < deadline and should_continue():
        candidates = sorted(glob.glob('/dev/ttyUSB*') + glob.glob('/dev/ttyACM*'))
        if logger is not None and not announced:
            logger.info(f"Autodetect ID:{expected_id} entre {candidates}")
            announced = True

        any_busy = False
        for port in candidates:
            if not should_continue():
                return '', None
            result = _probe_port(port, baud, expected_id, logger=logger)
            if isinstance(result, serial.Serial):
                if logger is not None:
                    logger.info(f"Match {port} -> ID:{expected_id}")
                return port, result
            if result is None:
                any_busy = True

        if any_busy:
            time.sleep(RETRY_BUSY_DELAY_S)
        else:
            time.sleep(1.0)

    return '', None


class Esp32Bridge(Node):
    def __init__(self):
        super().__init__('esp32_bridge')

        self.declare_parameter('port',         'auto')
        self.declare_parameter('expected_id',  '')
        self.declare_parameter('baud',         115200)
        self.declare_parameter('cmd_topic',    '/esp32_cmd')
        self.declare_parameter('status_topic', '/esp32_status')

        port_param   = self.get_parameter('port').get_parameter_value().string_value
        self._expected_id = self.get_parameter('expected_id').get_parameter_value().string_value
        baud         = self.get_parameter('baud').get_parameter_value().integer_value or 115200
        cmd_topic    = self.get_parameter('cmd_topic').get_parameter_value().string_value
        status_topic = self.get_parameter('status_topic').get_parameter_value().string_value

        # Estado
        self.ser = None
        self._running = True
        self._last_no_ser_warn = 0.0
        self._cmd_count_no_ser = 0

        # CRUCIAL: crear pub/sub ANTES de hacer autodetect.
        # Asi el resto de nodos siempre nos ven como subscriber valido y
        # los tests con `ros2 topic pub --once` no se quedan colgados.
        self.pub_status = self.create_publisher(String, status_topic, 50)
        self.create_subscription(String, cmd_topic, self._on_cmd, 50)

        self.get_logger().info(
            f"[{self.get_name()}] iniciado.  "
            f"cmd={cmd_topic}  status={status_topic}  "
            f"id={self._expected_id or '(no check)'}  "
            f"port={port_param}  -> conectando ESP32 en background...")

        # Hilo de RX (espera a que self.ser este listo)
        self._rx_thread = threading.Thread(target=self._rx_loop, daemon=True)
        self._rx_thread.start()

        # Hilo de inicializacion del puerto serie (autodetect o apertura
        # directa). No bloqueamos __init__ aqui.
        self._init_thread = threading.Thread(
            target=self._init_serial,
            args=(port_param, baud),
            daemon=True)
        self._init_thread.start()

    def _init_serial(self, port_param: str, baud: int):
        """Resuelve el puerto y abre el Serial. Corre en background."""
        if serial is None:
            self.get_logger().error("pyserial no esta instalado.")
            return

        p = (port_param or '').strip().lower()
        if p in ('', 'auto'):
            if not self._expected_id:
                self.get_logger().error(
                    "port=auto requiere expected_id (MOTORS o LIFT).")
                return
            port, ser = _autodetect_port(
                self._expected_id, baud,
                logger=self.get_logger(),
                should_continue=lambda: self._running)
            if ser is not None:
                ser.timeout = 0.1
                self.ser = ser
                self.get_logger().info(
                    f"[{self.get_name()}] PUERTO LISTO: {port}@{baud} "
                    f"id={self._expected_id}")
            else:
                self.get_logger().error(
                    f"No se encontro ninguna ESP32 con ID:{self._expected_id} "
                    f"despues de {AUTODETECT_BUDGET_S:.0f}s. "
                    f"Verifica: (1) la ESP32 esta conectada y alimentada, "
                    f"(2) el firmware tiene WHOAMI (firmware actualizado), "
                    f"(3) ModemManager esta deshabilitado "
                    f"(sudo systemctl disable ModemManager).")
        else:
            # Puerto explicito, sin verificacion de ID
            try:
                s = serial.Serial(port_param, baud, timeout=0.1)
                time.sleep(PROBE_CDC_RESET_S)
                self.ser = s
                self.get_logger().info(
                    f"[{self.get_name()}] PUERTO LISTO: {port_param}@{baud} "
                    f"(sin verificacion de ID)")
            except Exception as e:
                self.get_logger().error(
                    f"No se pudo abrir {port_param}: {e}")

    def _on_cmd(self, msg: String):
        if self.ser is None:
            self._cmd_count_no_ser += 1
            now = time.monotonic()
            if now - self._last_no_ser_warn > 5.0:
                self.get_logger().warn(
                    f"[{self.get_name()}] Comando recibido pero ESP32 "
                    f"todavia no esta lista ({self._cmd_count_no_ser} "
                    f"descartados). Ultimo: {msg.data}")
                self._last_no_ser_warn = now
                self._cmd_count_no_ser = 0
            return
        line = msg.data.strip() + '\n'
        try:
            self.ser.write(line.encode('ascii'))
        except Exception as e:
            self.get_logger().error(f"Error escribiendo serie: {e}")

    def _rx_loop(self):
        while self._running and rclpy.ok():
            if self.ser is None:
                time.sleep(0.5)
                continue
            try:
                raw = self.ser.readline()
                if not raw:
                    continue
                line = raw.decode('ascii', errors='ignore').strip()
                if not line:
                    continue
                self.pub_status.publish(String(data=line))
            except Exception as e:
                self.get_logger().warn(f"RX error: {e}")
                time.sleep(0.2)

    def destroy_node(self):
        self._running = False
        try:
            if self.ser is not None:
                self.ser.close()
        except Exception:
            pass
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = Esp32Bridge()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
