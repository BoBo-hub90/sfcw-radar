"""
Touchscreen display for the SFCW radar (white card theme).

Renders the live detection result on a WaveShare 5-inch HDMI touchscreen
(800x480) attached to a Raspberry Pi 4, using pygame in fullscreen.

The display runs its own daemon thread so the acquisition/processing loop is
never blocked by rendering. The latest pipeline result is handed over with
update() and read by the draw loop under a lock, so the two threads stay
consistent.

Layout (800x480, white background):
  - Top:    three stat cards (Detection Status, Detection Time, Signal)
  - Bottom: a live range-profile bar chart in dB

Usage:
    display = RadarDisplay()
    display.start()
    ...
    display.update(result)   # result = pipeline.run(...) dict
    ...
    display.stop()
"""

from __future__ import annotations

import threading
import time
from datetime import datetime

import numpy as np
import pygame

from utils.logger import get_logger

log = get_logger(__name__)

WIDTH, HEIGHT = 800, 480
FPS = 15  # modest refresh rate is plenty for a touchscreen

# --- Colors (RGB) ---
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GREY_LABEL = (120, 120, 120)     # card labels / axis text
GREY_BORDER = (204, 204, 204)    # #cccccc neutral card border
GRID_GREY = (220, 220, 220)      # horizontal grid lines
BLUE = (26, 95, 168)             # #1a5fa8 confidence text

DARK_RED = (139, 26, 26)         # #8b1a1a detected value
DARK_GREEN = (42, 107, 42)       # #2a6b2a no-target value
BORDER_RED = (192, 57, 43)       # #c0392b detected card border
BORDER_GREEN = (74, 124, 74)     # #4a7c4a no-target card border
BG_RED = (255, 245, 245)         # #fff5f5 detected card fill
BG_GREEN = (245, 255, 245)       # #f5fff5 no-target card fill

# Bar colors by amplitude band (dB).
BAR_LIGHT_PINK = (244, 179, 179)  # #f4b3b3  (< 9 dB)
BAR_SALMON = (229, 115, 115)      # #e57373  (9-12 dB)
BAR_RED = (192, 57, 43)           # #c0392b  (12-15 dB)
BAR_DARK_RED = (139, 26, 26)      # #8b1a1a  (> 15 dB)

# --- Card geometry: three cards across the top ---
CARD_W, CARD_H, CARD_Y = 240, 110, 20
CARD_GAP = 20
CARD_X = [20, 20 + CARD_W + CARD_GAP, 20 + 2 * (CARD_W + CARD_GAP)]  # 20,280,540

# --- Chart geometry ---
CHART = pygame.Rect(20, 150, 760, 280)
Y_DB_MAX = 20.0          # Y axis spans 0..20 dB
Y_TICK_STEP = 2.5        # label/grid every 2.5 dB
X_RANGE_MAX_M = 10.0     # X axis spans 0..10 m
X_TICK_STEP_M = 2.5      # label every 2.5 m

# Border width approximating the spec's 1.5 px (pygame needs integer widths).
BORDER_PX = 2

# --- Close button (top-right corner) ---
CLOSE_BTN = pygame.Rect(750, 10, 40, 30)
CLOSE_TEXT_COLOR = (80, 80, 80)  # dark grey "X" glyph


def _amplitude_to_db(values: np.ndarray) -> np.ndarray:
    """Convert linear magnitudes to dB via 20*log10(|x| + 1e-9)."""
    return 20.0 * np.log10(np.abs(values) + 1e-9)


def _bar_color(db: float):
    """Pick a bar color from its amplitude in dB."""
    if db < 9.0:
        return BAR_LIGHT_PINK
    if db < 12.0:
        return BAR_SALMON
    if db < 15.0:
        return BAR_RED
    return BAR_DARK_RED


class RadarDisplay:
    """Fullscreen pygame UI (white card theme) for live SFCW results."""

    def __init__(self):
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._running = False

        # Shared state, guarded by _lock.
        self._result: dict | None = None
        self._last_update_ts: float | None = None
        # Timestamp when the current continuous detection started (None if idle).
        self._detect_start_ts: float | None = None

        # Created inside the draw thread (pygame must init there).
        self._screen: pygame.Surface | None = None
        self._fonts: dict[str, pygame.font.Font] = {}

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def start(self) -> None:
        """Launch the display loop in a separate daemon thread."""
        if self._thread is not None and self._thread.is_alive():
            log.warning("Display already running")
            return
        self._running = True
        self._thread = threading.Thread(
            target=self._run, name="RadarDisplay", daemon=True
        )
        self._thread.start()
        log.info("Display thread started")

    def update(self, result: dict) -> None:
        """
        Hand the latest pipeline result to the display (thread-safe).

        Also maintains the continuous-detection start time so the Detection Time
        card can show how long a target has been present.

        Args:
            result: dict with keys detected (bool), target_range_m (float),
                range_profile (np.ndarray), cfar_threshold (np.ndarray).
        """
        with self._lock:
            prev_detected = bool(self._result["detected"]) if self._result else False
            now_detected = bool(result["detected"])
            if now_detected and not prev_detected:
                self._detect_start_ts = time.time()  # rising edge
            elif not now_detected:
                self._detect_start_ts = None          # reset when idle

            self._result = result
            self._last_update_ts = time.time()

    def stop(self) -> None:
        """Signal the display loop to exit and tear down pygame cleanly."""
        with self._lock:
            self._running = False
        t = self._thread
        if t is not None and threading.current_thread() is not t:
            t.join(timeout=2.0)

    def is_running(self) -> bool:
        """Return True while the display loop is active."""
        with self._lock:
            return self._running

    # ------------------------------------------------------------------ #
    # Draw thread
    # ------------------------------------------------------------------ #

    def _run(self) -> None:
        """Thread entry point: init pygame, run the draw loop, then quit."""
        try:
            pygame.init()
            pygame.display.set_caption("SFCW Radar")
            self._screen = pygame.display.set_mode(
                (WIDTH, HEIGHT), pygame.FULLSCREEN
            )
            pygame.mouse.set_visible(False)
            self._fonts = {
                "card_label": self._font(18, bold=True),
                "card_value": self._font(34, bold=True),
                "card_sub": self._font(17),
                "title": self._font(22, bold=True),
                "axis": self._font(16),
            }
            clock = pygame.time.Clock()

            while self.is_running():
                self._handle_events()
                result, detect_start = self._snapshot()
                self._draw(result, detect_start)
                pygame.display.flip()
                clock.tick(FPS)
        except Exception as e:  # keep the radar alive if the UI fails
            log.error("Display loop crashed: %s", e)
        finally:
            pygame.quit()
            log.info("Display thread stopped")

    @staticmethod
    def _font(size: int, bold: bool = False) -> pygame.font.Font:
        """Build a default pygame font at the given size, optionally bold."""
        font = pygame.font.Font(None, size)
        font.set_bold(bold)
        return font

    def _handle_events(self) -> None:
        """Process pygame events (window close / close-button tap stop the display)."""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.stop()
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if CLOSE_BTN.collidepoint(event.pos):
                    self.stop()

    def _snapshot(self) -> tuple[dict | None, float | None]:
        """Return a consistent copy of the shared state under the lock."""
        with self._lock:
            return self._result, self._detect_start_ts

    # ------------------------------------------------------------------ #
    # Rendering
    # ------------------------------------------------------------------ #

    def _draw(self, result: dict | None, detect_start: float | None) -> None:
        """Render one full frame."""
        self._screen.fill(WHITE)
        self._draw_status_card(result)
        self._draw_time_card(result, detect_start)
        self._draw_signal_card(result)
        self._draw_chart(result)
        self._draw_close_button()

    def _blit_text(self, text, font_key, color, pos) -> None:
        """Render text and blit it at the given top-left position."""
        surf = self._fonts[font_key].render(text, True, color)
        self._screen.blit(surf, pos)

    def _draw_card_frame(self, x, fill, border) -> pygame.Rect:
        """Draw a card background + border, return its rect."""
        rect = pygame.Rect(x, CARD_Y, CARD_W, CARD_H)
        pygame.draw.rect(self._screen, fill, rect)
        pygame.draw.rect(self._screen, border, rect, BORDER_PX)
        return rect

    def _draw_status_card(self, result: dict | None) -> None:
        """Card 1 — detection state, color-coded red (target) / green (clear)."""
        detected = bool(result["detected"]) if result else False

        fill = BG_RED if detected else BG_GREEN
        border = BORDER_RED if detected else BORDER_GREEN
        value_color = DARK_RED if detected else DARK_GREEN
        value_text = "TARGET DETECTED" if detected else "NO TARGET"
        sub_text = "Motion above threshold" if detected else "Scene appears stable"

        rect = self._draw_card_frame(CARD_X[0], fill, border)
        self._blit_text("DETECTION STATUS", "card_label", GREY_LABEL,
                        (rect.x + 12, rect.y + 10))
        self._blit_text(value_text, "card_value", value_color,
                        (rect.x + 12, rect.y + 40))
        self._blit_text(sub_text, "card_sub", value_color,
                        (rect.x + 12, rect.y + 80))

    def _draw_time_card(self, result: dict | None, detect_start: float | None) -> None:
        """Card 2 — how long the current detection has been continuously active."""
        detected = bool(result["detected"]) if result else False
        elapsed = (time.time() - detect_start) if detect_start else 0.0
        sub_text = "active detection" if detected else "no active detection"

        rect = self._draw_card_frame(CARD_X[1], WHITE, GREY_BORDER)
        self._blit_text("DETECTION TIME", "card_label", GREY_LABEL,
                        (rect.x + 12, rect.y + 10))
        self._blit_text(f"{elapsed:.1f} s", "card_value", BLACK,
                        (rect.x + 12, rect.y + 40))
        self._blit_text(sub_text, "card_sub", GREY_LABEL,
                        (rect.x + 12, rect.y + 80))

    def _draw_signal_card(self, result: dict | None) -> None:
        """Card 3 — CFAR peak amplitude in dB and a coarse confidence label."""
        peak_db = 0.0
        if result is not None:
            profile = np.asarray(result["range_profile"], dtype=float)
            if profile.size:
                peak_db = float(_amplitude_to_db(np.max(profile)))

        if peak_db >= 12.0:
            confidence = "High"
        elif peak_db >= 9.0:
            confidence = "Medium"
        else:
            confidence = "Low"

        rect = self._draw_card_frame(CARD_X[2], WHITE, GREY_BORDER)
        self._blit_text("SIGNAL", "card_label", GREY_LABEL,
                        (rect.x + 12, rect.y + 10))
        self._blit_text(f"{peak_db:.1f} dB", "card_value", BLACK,
                        (rect.x + 12, rect.y + 40))
        self._blit_text(f"Confidence: {confidence}", "card_sub", BLUE,
                        (rect.x + 12, rect.y + 80))

    def _draw_chart(self, result: dict | None) -> None:
        """Bottom — live range-profile bar chart with dB grid and axes."""
        self._blit_text("Live range profile", "title", BLACK,
                        (CHART.x + 4, CHART.y))

        # Inner plotting area: room for the title, y labels, and x labels.
        plot = pygame.Rect(
            CHART.x + 45,
            CHART.y + 26,
            CHART.w - 45 - 12,
            CHART.h - 26 - 24,
        )

        # Horizontal grid + Y tick labels (0..20 dB every 2.5).
        n_ticks = int(round(Y_DB_MAX / Y_TICK_STEP))
        for k in range(n_ticks + 1):
            db = k * Y_TICK_STEP
            y = plot.bottom - int(round((db / Y_DB_MAX) * plot.h))
            pygame.draw.line(self._screen, GRID_GREY,
                             (plot.x, y), (plot.right, y), 1)
            label = self._fonts["axis"].render(f"{db:.1f}", True, GREY_LABEL)
            self._screen.blit(
                label, (plot.x - label.get_width() - 6, y - label.get_height() // 2)
            )

        # X tick labels (0..10 m every 2.5).
        n_xticks = int(round(X_RANGE_MAX_M / X_TICK_STEP_M))
        for k in range(n_xticks + 1):
            m = k * X_TICK_STEP_M
            x = plot.x + int(round((m / X_RANGE_MAX_M) * plot.w))
            label = self._fonts["axis"].render(f"{m:.1f}", True, GREY_LABEL)
            self._screen.blit(label, (x - label.get_width() // 2, plot.bottom + 4))

        # Vertical "dB" axis label on the far left, rotated 90 degrees.
        db_label = self._fonts["axis"].render("dB", True, GREY_LABEL)
        db_label = pygame.transform.rotate(db_label, 90)
        self._screen.blit(
            db_label,
            (CHART.x, plot.centery - db_label.get_height() // 2),
        )

        # Bars: one per displayed range bin, colored by dB band. For a cleaner
        # look the profile is downsampled to every 2nd bin (display only), bars
        # are drawn narrower than their slot to leave gaps, and corners are
        # slightly rounded.
        if result is not None:
            profile = np.asarray(result["range_profile"], dtype=float)
            # Downsample for display only (101 bars instead of 201).
            profile_disp = profile[::2]
            n_disp = profile_disp.size
            if n_disp > 0:
                profile_db = np.clip(
                    _amplitude_to_db(profile_disp), 0.0, Y_DB_MAX
                )

                # Slot = full width / bars; bar fills 60% of it (40% gap).
                slot_w = plot.w / n_disp
                bar_w = max(1, int(round(slot_w * 0.6)))
                radius = min(2, bar_w // 2)

                # Highlight the peak bar in green when a target is detected.
                detected = bool(result["detected"])
                peak_idx = int(np.argmax(profile_db)) if n_disp else -1

                for i in range(n_disp):
                    db = float(profile_db[i])
                    h = int(round((db / Y_DB_MAX) * plot.h))
                    # Center the bar within its slot to balance the gaps.
                    slot_left = plot.x + i * slot_w
                    x0 = int(round(slot_left + (slot_w - bar_w) / 2))
                    bar = pygame.Rect(x0, plot.bottom - h, bar_w, h)

                    color = _bar_color(db)
                    if detected and i == peak_idx:
                        color = DARK_GREEN  # #2a6b2a peak highlight
                    pygame.draw.rect(
                        self._screen, color, bar, border_radius=radius
                    )

        # Threshold reference line at 10 dB: a light-grey dashed horizontal line
        # drawn over the bars (alternating filled/empty segments).
        y_thr = plot.bottom - int(round((10.0 / Y_DB_MAX) * plot.h))
        dash, gap = 6, 4
        x = plot.x
        while x < plot.right:
            x_end = min(x + dash, plot.right)
            pygame.draw.line(self._screen, GREY_BORDER,
                             (x, y_thr), (x_end, y_thr), 1)
            x += dash + gap

        # Axes border last so it frames the bars cleanly.
        pygame.draw.rect(self._screen, BLACK, plot, BORDER_PX)

    def _draw_close_button(self) -> None:
        """Draw the top-right close button on top of everything else."""
        pygame.draw.rect(self._screen, WHITE, CLOSE_BTN)
        pygame.draw.rect(self._screen, GREY_BORDER, CLOSE_BTN, BORDER_PX)
        glyph = self._fonts["card_label"].render("✕", True, CLOSE_TEXT_COLOR)
        self._screen.blit(
            glyph,
            (CLOSE_BTN.centerx - glyph.get_width() // 2,
             CLOSE_BTN.centery - glyph.get_height() // 2),
        )
