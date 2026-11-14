#!/usr/bin/env python3
"""Route tracker for an MPU6050 + buzzer + OLED on a Raspberry Pi.

Counts footsteps (accelerometer) and tracks which way you turn (gyro) to
estimate where you are, compares that with the planned route, sounds the
buzzer when you drift off it, and shows progress on the OLED screen.

  GET  /steps  -> position, distance along the route, off_route flag
  GET  /reset  -> back to the start of the route
  GET  /beep   -> one short beep, to test the buzzer wiring
  POST /route  -> {"points": [[x, y], ...]} in metres, replaces the route

Start standing still at the first route point, facing along the first leg.

Run:  python3 steps.py
"""
import json
import math
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

STRIDE_M = 0.7        # metres per step; measure yours and change it
THRESHOLD_G = 0.25    # how hard a jolt counts as a step; raise if it over-counts
MIN_GAP_S = 0.3       # ignore a second spike this soon after a step
OFF_ROUTE_M = 1.5     # buzz when this far from the route line
LOOKAHEAD_M = 1.5     # the arrow aims at the route this far ahead of you
SAMPLE_HZ = 50
PORT = 8000

BUZZER_GPIO = 5       # physical pin 29
BUZZER_HZ = 2000

# Route used until the app POSTs a real one: 5 m ahead, turn left, 5 m.
DEFAULT_ROUTE = [(0.0, 0.0), (5.0, 0.0), (5.0, 5.0)]

OLED_ADDR = 0x3C
FONT_PATH = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"

MPU_ADDR = 0x68
PWR_MGMT_1 = 0x6B
GYRO_CONFIG = 0x1B
ACCEL_XOUT_H = 0x3B
ACCEL_LSB_PER_G = 16384.0   # +/-2 g
GYRO_LSB_PER_DPS = 65.5     # +/-500 deg/s


class StepDetector:
    """Counts a step each time acceleration spikes above the resting level."""

    def __init__(self, threshold=THRESHOLD_G, min_gap=MIN_GAP_S):
        self.threshold = threshold
        self.min_gap = min_gap
        self.baseline = 1.0   # slow average, tracks gravity (about 1 g)
        self.armed = True     # must drop back down before the next step counts
        self.last_step = -1e9

    def update(self, magnitude_g, now):
        self.baseline += 0.02 * (magnitude_g - self.baseline)
        delta = magnitude_g - self.baseline
        if self.armed and delta > self.threshold and now - self.last_step > self.min_gap:
            self.last_step = now
            self.armed = False
            return True
        if delta < self.threshold / 2:
            self.armed = True
        return False


class Tracker:
    """Dead reckoning: each step moves the position one stride along the heading."""

    def __init__(self, route=DEFAULT_ROUTE, stride=STRIDE_M, tolerance=OFF_ROUTE_M):
        self.stride = stride
        self.tolerance = tolerance
        self.set_route(route)

    def set_route(self, points):
        points = [(float(x), float(y)) for x, y in points]
        if len(points) < 2 or points[0] == points[1]:
            raise ValueError("route needs at least two different points")
        self.route = points
        self.total = sum(math.dist(a, b) for a, b in zip(points, points[1:]))
        self.reset()

    def reset(self):
        (x0, y0), (x1, y1) = self.route[0], self.route[1]
        self.x, self.y = x0, y0
        self.heading = math.atan2(y1 - y0, x1 - x0)  # facing along the first leg
        self.steps = 0
        self.along = 0.0
        self.cross_track = 0.0
        self.off_route = False

    def turn(self, yaw_rate_dps, dt):
        """Positive rate = turning left (anticlockwise seen from above)."""
        self.heading += math.radians(yaw_rate_dps) * dt

    def step(self):
        self.steps += 1
        self.x += self.stride * math.cos(self.heading)
        self.y += self.stride * math.sin(self.heading)
        self._locate()

    def _locate(self):
        """Find the nearest point on the route line to the current position."""
        best = (float("inf"), 0.0)
        walked = 0.0
        for (ax, ay), (bx, by) in zip(self.route, self.route[1:]):
            length = math.hypot(bx - ax, by - ay)
            if length == 0:
                continue
            t = ((self.x - ax) * (bx - ax) + (self.y - ay) * (by - ay)) / length ** 2
            t = max(0.0, min(1.0, t))
            d = math.hypot(self.x - (ax + t * (bx - ax)), self.y - (ay + t * (by - ay)))
            if d < best[0]:
                best = (d, walked + t * length)
            walked += length
        self.cross_track, self.along = best
        self.off_route = self.cross_track > self.tolerance

    def _point_at(self, distance):
        """The point this many metres along the route (clamped to its end)."""
        for (ax, ay), (bx, by) in zip(self.route, self.route[1:]):
            length = math.hypot(bx - ax, by - ay)
            if distance <= length and length > 0:
                t = distance / length
                return ax + t * (bx - ax), ay + t * (by - ay)
            distance -= length
        return self.route[-1]

    def turn_deg(self):
        """Which way to go, relative to the way you're facing. 0 = straight
        ahead, positive = to your left, negative = to your right."""
        tx, ty = self._point_at(self.along + LOOKAHEAD_M)
        if math.hypot(tx - self.x, ty - self.y) < 0.3:
            return 0.0
        diff = math.degrees(math.atan2(ty - self.y, tx - self.x) - self.heading)
        return (diff + 180) % 360 - 180

    def status(self):
        return {
            "steps": self.steps,
            "distance_m": round(self.steps * self.stride, 2),
            "x": round(self.x, 2),
            "y": round(self.y, 2),
            "heading_deg": round(math.degrees(self.heading) % 360, 1),
            "along_m": round(self.along, 2),
            "route_m": round(self.total, 2),
            "cross_track_m": round(self.cross_track, 2),
            "off_route": self.off_route,
            "turn_deg": round(self.turn_deg(), 1),
            "arrived": self.along >= self.total - self.stride and not self.off_route,
        }


detector = StepDetector()
tracker = Tracker()
lock = threading.Lock()
beep_request = threading.Event()


def to_signed(high, low):
    value = (high << 8) | low
    return value - 65536 if value & 0x8000 else value


def read_imu(bus):
    """Returns (accel in g, gyro in deg/s), each an (x, y, z) tuple."""
    d = bus.read_i2c_block_data(MPU_ADDR, ACCEL_XOUT_H, 14)
    accel = tuple(to_signed(d[i], d[i + 1]) / ACCEL_LSB_PER_G for i in (0, 2, 4))
    gyro = tuple(to_signed(d[i], d[i + 1]) / GYRO_LSB_PER_DPS for i in (8, 10, 12))
    return accel, gyro


def open_bus():
    try:
        from smbus2 import SMBus
    except ImportError:
        from smbus import SMBus
    return SMBus(1)


def sensor_loop():
    bus = open_bus()
    bus.write_byte_data(MPU_ADDR, PWR_MGMT_1, 0)      # wake the sensor up
    bus.write_byte_data(MPU_ADDR, GYRO_CONFIG, 0x08)  # +/-500 deg/s
    time.sleep(0.1)

    # The gyro reads slightly non-zero when still; measure that and subtract it.
    print("Calibrating: hold still for 2 seconds...", flush=True)
    samples = [read_imu(bus) for _ in range(100) if not time.sleep(0.02)]
    bias = [sum(g[i] for _, g in samples) / len(samples) for i in range(3)]
    up = [sum(a[i] for a, _ in samples) / len(samples) for i in range(3)]
    print("Ready. Walk.", flush=True)

    period = 1.0 / SAMPLE_HZ
    last = time.monotonic()
    while True:
        try:
            accel, gyro = read_imu(bus)
        except OSError:
            time.sleep(0.1)  # loose wire or a glitch: try again
            last = time.monotonic()
            continue
        now = time.monotonic()
        dt, last = now - last, now

        # "Up" is wherever gravity points, so the sensor can sit at any angle.
        up = [u + 0.02 * (a - u) for u, a in zip(up, accel)]
        norm = math.sqrt(sum(u * u for u in up)) or 1.0
        yaw_rate = sum((g - b) * u / norm for g, b, u in zip(gyro, bias, up))
        if abs(yaw_rate) < 0.5:
            yaw_rate = 0.0  # ignore sensor noise so the heading doesn't creep

        magnitude = math.sqrt(sum(a * a for a in accel))
        with lock:
            tracker.turn(yaw_rate, dt)
            stepped = detector.update(magnitude, now)
            if stepped:
                tracker.step()
            s = tracker.status()
        if stepped:
            flag = "  OFF ROUTE" if s["off_route"] else ""
            print(f"step {s['steps']}  along {s['along_m']} m  "
                  f"off by {s['cross_track_m']} m{flag}", flush=True)
        time.sleep(period)


def buzzer_loop():
    """Beeps while off route. The module is low-level trigger, hence active_high=False."""
    try:
        from gpiozero import PWMOutputDevice
        buzzer = PWMOutputDevice(BUZZER_GPIO, active_high=False,
                                 initial_value=0, frequency=BUZZER_HZ)
    except Exception as e:  # no gpiozero, or not on a Pi
        print(f"Buzzer disabled: {e}", flush=True)
        return
    while True:
        with lock:
            off = tracker.off_route
        if off or beep_request.is_set():
            beep_request.clear()
            buzzer.value = 0.5   # half-on square wave = a tone
            time.sleep(0.15)
            buzzer.value = 0
            time.sleep(0.15)
        else:
            time.sleep(0.05)


class Oled:
    """Minimal driver for a 128x64 SSD1306 OLED on I2C."""

    INIT = (0xAE, 0xD5, 0x80, 0xA8, 0x3F, 0xD3, 0x00, 0x40, 0x8D, 0x14,
            0x20, 0x00, 0xA1, 0xC8, 0xDA, 0x12, 0x81, 0xCF, 0xD9, 0xF1,
            0xDB, 0x40, 0xA4, 0xA6, 0xAF)

    def __init__(self, bus, addr=OLED_ADDR):
        self.bus, self.addr = bus, addr
        for c in self.INIT:
            self.command(c)

    def _write(self, fn, *args):
        """I2C over jumper wires drops the odd message, so try a few times."""
        for attempt in range(5):
            try:
                return fn(*args)
            except OSError:
                if attempt == 4:
                    raise
                time.sleep(0.01)

    def command(self, c):
        self._write(self.bus.write_byte_data, self.addr, 0x00, c)

    def show(self, image):
        """Send a 128x64 1-bit PIL image to the screen."""
        px = image.load()
        data = []
        for page in range(8):          # the screen is 8 strips, each 8 pixels tall
            for x in range(128):
                byte = 0
                for bit in range(8):
                    if px[x, page * 8 + bit]:
                        byte |= 1 << bit
                data.append(byte)
        for c in (0x21, 0, 127, 0x22, 0, 7):
            self.command(c)
        for i in range(0, len(data), 32):
            self._write(self.bus.write_i2c_block_data, self.addr, 0x40, data[i:i + 32])


def load_font(size):
    from PIL import ImageFont
    try:
        return ImageFont.truetype(FONT_PATH, size)
    except OSError:
        return ImageFont.load_default()


def draw_arrow(d, cx, cy, turn_deg, ink=1, length=17):
    """Arrow centred on (cx, cy). Up = straight ahead, left = turn left."""
    a = math.radians(turn_deg)
    dx, dy = -math.sin(a), -math.cos(a)   # unit vector the arrow points along
    px, py = -dy, dx                       # at right angles to it
    tip = (cx + length * dx, cy + length * dy)
    tail = (cx - length * dx, cy - length * dy)
    neck = (cx + (length - 12) * dx, cy + (length - 12) * dy)
    d.line((tail, neck), fill=ink, width=5)
    d.polygon((tip,
               (neck[0] + 10 * px, neck[1] + 10 * py),
               (neck[0] - 10 * px, neck[1] - 10 * py)), fill=ink)


def render(s, big, small, flash=False):
    """Draw the status dict from Tracker.status() as a 128x64 image.
    Arrow on the left (hold the screen with its top pointing forwards),
    text on the right, progress bar along the bottom."""
    from PIL import Image, ImageDraw
    img = Image.new("1", (128, 64), 0)
    d = ImageDraw.Draw(img)
    off = s["off_route"]
    ink = 1
    if off and flash:
        d.rectangle((0, 0, 127, 63), fill=1)
        ink = 0

    if s["arrived"]:
        d.ellipse((10, 10, 34, 34), outline=ink, width=3)
        d.ellipse((18, 18, 26, 26), fill=ink)
    else:
        draw_arrow(d, 22, 22, s["turn_deg"], ink)

    def centred(y, text, font):
        x = max(44, 44 + (84 - d.textlength(text, font=font)) / 2)
        d.text((x, y), text, font=font, fill=ink)

    if off:
        centred(3, "OFF ROUTE", big)
        centred(24, f"{s['cross_track_m']:.1f} m off", small)
    else:
        centred(3, "ARRIVED" if s["arrived"] else "ON ROUTE", big)
        centred(24, f"{s['along_m']:.1f}/{s['route_m']:.1f} m", small)

    d.rectangle((0, 50, 127, 63), outline=ink)
    filled = int(124 * min(1.0, s["along_m"] / s["route_m"]))
    if filled:
        d.rectangle((2, 52, 1 + filled, 61), fill=ink)
    return img


def display_loop():
    """Redraws the OLED four times a second. Keeps retrying if it drops out."""
    try:
        import PIL  # noqa: F401  (fail early if Pillow is missing)
        big, small = load_font(12), load_font(11)
    except Exception as e:
        print(f"Display disabled: {e}", flush=True)
        return
    oled = None
    state = "starting"
    flash = False
    while True:
        try:
            if oled is None:
                oled = Oled(open_bus())
            with lock:
                s = tracker.status()
            flash = not flash
            oled.show(render(s, big, small, flash))
            if state != "ok":
                print("Display running.", flush=True)
                state = "ok"
        except OSError as e:
            if state != "failing":
                print(f"Display not responding ({e}); will keep trying.", flush=True)
                state = "failing"
            oled = None
            time.sleep(1)
            continue
        time.sleep(0.25)


class Handler(BaseHTTPRequestHandler):
    def _send(self, code, payload):
        body = json.dumps(payload).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self):
        self._send(200, {})

    def do_GET(self):
        if self.path.startswith("/reset"):
            with lock:
                tracker.reset()
        elif self.path.startswith("/beep"):
            beep_request.set()
        elif not self.path.startswith("/steps"):
            self._send(404, {"error": "not found"})
            return
        with lock:
            self._send(200, tracker.status())

    def do_POST(self):
        if not self.path.startswith("/route"):
            self._send(404, {"error": "not found"})
            return
        try:
            length = int(self.headers.get("Content-Length", 0))
            points = json.loads(self.rfile.read(length))["points"]
            with lock:
                tracker.set_route(points)
                status = tracker.status()
        except (ValueError, KeyError, TypeError) as e:
            self._send(400, {"error": str(e)})
            return
        self._send(200, status)

    def log_message(self, *args):
        pass  # keep the terminal clear for step prints


if __name__ == "__main__":
    threading.Thread(target=sensor_loop, daemon=True).start()
    threading.Thread(target=buzzer_loop, daemon=True).start()
    threading.Thread(target=display_loop, daemon=True).start()
    print(f"Serving on http://<pi-address>:{PORT}/steps", flush=True)
    ThreadingHTTPServer(("0.0.0.0", PORT), Handler).serve_forever()
