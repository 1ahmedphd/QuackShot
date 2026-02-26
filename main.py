import pygame
import cv2
import numpy as np
import mediapipe as mp
from collections import deque

from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from enum import IntEnum

class HandLandmark(IntEnum):
    WRIST = 0
    THUMB_CMC = 1
    THUMB_MCP = 2
    THUMB_IP = 3
    THUMB_TIP = 4
    INDEX_FINGER_MCP = 5
    INDEX_FINGER_PIP = 6
    INDEX_FINGER_DIP = 7
    INDEX_FINGER_TIP = 8
    MIDDLE_FINGER_MCP = 9
    MIDDLE_FINGER_PIP = 10
    MIDDLE_FINGER_DIP = 11
    MIDDLE_FINGER_TIP = 12
    RING_FINGER_MCP = 13
    RING_FINGER_PIP = 14
    RING_FINGER_DIP = 15
    RING_FINGER_TIP = 16
    PINKY_MCP = 17
    PINKY_PIP = 18
    PINKY_DIP = 19
    PINKY_TIP = 20

# Screen dimensions
SCREEN_WIDTH = 1280
SCREEN_HEIGHT = 720
SCREEN_TITLE = "QuackShot"

# Game States
GAME_STATE_CALIBRATION = 0
GAME_STATE_PLAYING = 1

# Calibration Settings
CALIBRATION_TARGET_POSITIONS = [
    (SCREEN_WIDTH * 0.1, SCREEN_HEIGHT * 0.1),
    (SCREEN_WIDTH * 0.9, SCREEN_HEIGHT * 0.1),
    (SCREEN_WIDTH * 0.1, SCREEN_HEIGHT * 0.9),
    (SCREEN_WIDTH * 0.9, SCREEN_HEIGHT * 0.9),
    (SCREEN_WIDTH * 0.5, SCREEN_HEIGHT * 0.5)
]
CALIBRATION_HOLD_TIME = 2.0  # seconds to hold gesture at target
CALIBRATION_POINT_COOLDOWN = 1.0 # seconds cooldown after a point is registered

# Game Constants
DUCK_SPAWN_INTERVAL = 1.5 # seconds between duck spawns

# Set True to show live gesture debug values on screen
DEBUG_GESTURES = True

class Game:
    def __init__(self):
        pygame.init()
        # Using RESIZABLE and SCALED for a "full window" feel that can be maximized
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT), pygame.RESIZABLE | pygame.SCALED)
        pygame.display.set_caption(SCREEN_TITLE)
        self.clock = pygame.time.Clock()

        # Load game assets
        self.background_image = pygame.image.load("background.png").convert()
        self.target_image = pygame.image.load("target.png").convert_alpha()
        self.duck_image = pygame.image.load("duck.png").convert_alpha()

        # Game State
        self.game_state = GAME_STATE_CALIBRATION

        # Calibration Variables
        self.calibration_data = []
        self.current_calibration_target_index = 0
        self.calibration_timer = 0.0
        self.calibration_cooldown_timer = 0.0

        # Game Cursor (after calibration)
        self.cursor_x = SCREEN_WIDTH // 2
        self.cursor_y = SCREEN_HEIGHT // 2

        # Shooting related variables
        self.is_shooting = False
        self.last_shot_time = 0.0
        self.SHOT_COOLDOWN = 0.5

        # Game elements
        self.ducks = pygame.sprite.Group()
        self.score = 0
        self.duck_spawn_timer = 0.0

        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            print("Error: Could not open video stream. Please ensure a webcam is connected and not in use by another application.")
            pygame.quit()
            exit()
            return
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, SCREEN_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, SCREEN_HEIGHT)

        # Store the last frame from the webcam
        self.camera_frame = None
        self.camera_surface = None

        # Initialize MediaPipe HandLandmarker
        base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')
        options = vision.HandLandmarkerOptions(base_options=base_options,
                                               num_hands=1,
                                               running_mode=vision.RunningMode.IMAGE)
        self.landmarker = vision.HandLandmarker.create_from_options(options)

        self.hand_landmarks = None
        self.mp_hands_landmark = HandLandmark

        self.running = True

        # Gesture smoothing
        self.finger_gun_history = deque(maxlen=10)
        self.gesture_debug_info = {}

    def _dist3d(self, a, b):
        """Euclidean distance between two MediaPipe landmarks in 3D."""
        return np.sqrt((a.x-b.x)**2 + (a.y-b.y)**2 + (a.z-b.z)**2)

    def _is_curl(self, lm, tip_idx, pip_idx, mcp_idx):
        """
        A finger is CURLED when its tip is close to its own MCP base knuckle.
        tip-to-MCP distance < pip-to-MCP distance * margin.
        Returns (is_curled, tip_to_mcp, pip_to_mcp) for debug.
        """
        mcp     = lm[mcp_idx]
        tip_d   = self._dist3d(lm[tip_idx], mcp)
        pip_d   = self._dist3d(lm[pip_idx], mcp)
        is_curl = tip_d < pip_d * 2.2
        return is_curl, round(tip_d, 3), round(pip_d, 3)

    def _is_wrist_extended(self, lm, tip_idx, mcp_idx):
        """
        A finger is EXTENDED when its tip is farther from the wrist than its MCP.
        Returns (is_extended, tip_dist, mcp_dist) for debug.
        """
        wrist   = lm[self.mp_hands_landmark.WRIST]
        tip_d   = self._dist3d(lm[tip_idx], wrist)
        mcp_d   = self._dist3d(lm[mcp_idx], wrist)
        is_ext  = tip_d > mcp_d * 1.05
        return is_ext, round(tip_d, 3), round(mcp_d, 3)

    def _is_finger_gun_raw(self, hand_landmarks):
        """
        Hybrid detection:
          - INDEX + THUMB  -> wrist-extension check
          - MIDDLE/RING/PINKY -> intra-finger curl check
        """
        if not hand_landmarks:
            return False

        lm = hand_landmarks
        H  = self.mp_hands_landmark

        index_ext, i_tip, i_mcp = self._is_wrist_extended(lm, H.INDEX_FINGER_TIP, H.INDEX_FINGER_MCP)
        thumb_ext, t_tip, t_mcp = self._is_wrist_extended(lm, H.THUMB_TIP,        H.THUMB_CMC)

        mid_curl, m_tip, m_pip = self._is_curl(lm, H.MIDDLE_FINGER_TIP, H.MIDDLE_FINGER_PIP, H.MIDDLE_FINGER_MCP)
        rng_curl, r_tip, r_pip = self._is_curl(lm, H.RING_FINGER_TIP,   H.RING_FINGER_PIP,   H.RING_FINGER_MCP)
        pky_curl, p_tip, p_pip = self._is_curl(lm, H.PINKY_TIP,         H.PINKY_PIP,         H.PINKY_MCP)

        self.gesture_debug_info = {
            "index":  (index_ext, i_tip, i_mcp, "ext",  "wrist"),
            "thumb":  (thumb_ext, t_tip, t_mcp, "ext",  "wrist"),
            "middle": (mid_curl,  m_tip, m_pip, "curl", "mcp"),
            "ring":   (rng_curl,  r_tip, r_pip, "curl", "mcp"),
            "pinky":  (pky_curl,  p_tip, p_pip, "curl", "mcp"),
        }

        return index_ext and thumb_ext and mid_curl and rng_curl and pky_curl

    def _is_finger_gun(self, hand_landmarks):
        """Smoothed over 10 frames, needs 60% agreement."""
        raw    = self._is_finger_gun_raw(hand_landmarks)
        self.finger_gun_history.append(raw)
        votes  = sum(self.finger_gun_history)
        is_gun = votes >= len(self.finger_gun_history) * 0.5
        if self.gesture_debug_info:
            self.gesture_debug_info["votes"] = f"{votes}/{len(self.finger_gun_history)}"
            self.gesture_debug_info["GUN"]   = is_gun
        return is_gun

    def _is_thumb_lowered(self, hand_landmarks):
        """Trigger: thumb curls inward."""
        if not hand_landmarks:
            return False
        lm = hand_landmarks
        H  = self.mp_hands_landmark
        is_curl, tip_d, pip_d = self._is_curl(lm, H.THUMB_TIP, H.THUMB_IP, H.THUMB_MCP)
        if self.gesture_debug_info:
            self.gesture_debug_info["thumb_trigger"] = (is_curl, tip_d, pip_d)
            self.gesture_debug_info["TRIGGER"]       = is_curl
        return is_curl

    def _draw_debug_overlay(self):
        """Live debug panel — shows per-finger distances and pass/fail status."""
        d = self.gesture_debug_info
        if not d:
            return
        font  = pygame.font.Font(None, 22)
        order = ["index", "thumb", "middle", "ring", "pinky"]
        lines = ["=== GESTURE DEBUG ==="]
        for name in order:
            if name not in d:
                continue
            ok, a, b, want, ref = d[name]
            goal   = "want CURL" if want == "curl" else "want EXT "
            status = "OK" if ok else "!!"
            lines.append(f"{name:<6} {a:.3f} vs {b:.3f} ({ref}) {goal} [{status}]")
        if "thumb_trigger" in d:
            trig, t, p = d["thumb_trigger"]
            lines.append(f"trigr  {t:.3f} vs {p:.3f} (mcp) want CURL [{'FIRE' if trig else '----'}]")
        lines.append(f"votes : {d.get('votes','?')}")
        lines.append(f"GUN   : {'*** YES ***' if d.get('GUN') else 'no'}")

        box_w = 360
        box_h = len(lines) * 20 + 10
        surf  = pygame.Surface((box_w, box_h), pygame.SRCALPHA)
        surf.fill((0, 0, 0, 170))
        self.screen.blit(surf, (10, 10))
        for i, line in enumerate(lines):
            if "YES" in line or "OK" in line or "FIRE" in line:
                color = (0, 255, 100)
            elif "!!" in line or line.strip().endswith("no") or "----" in line:
                color = (255, 80, 80)
            else:
                color = (210, 210, 210)
            self.screen.blit(font.render(line, True, color), (15, 15 + i * 20))

    def _spawn_duck(self):
        duck = Duck(self.duck_image, 0.5)

        if np.random.rand() > 0.5:
            duck.rect.x = SCREEN_WIDTH + duck.rect.width // 2
            duck.change_x = -np.random.randint(2, 6)
        else:
            duck.rect.x = -duck.rect.width // 2
            duck.change_x = np.random.randint(2, 6)

        duck.rect.y = np.random.randint(int(SCREEN_HEIGHT * 0.2), int(SCREEN_HEIGHT * 0.8))
        self.ducks.add(duck)

    def _map_hand_to_screen(self, hand_x, hand_y):
        if not self.calibration_data:
            return SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2

        min_hand_x = min(d['hand_x'] for d in self.calibration_data)
        max_hand_x = max(d['hand_x'] for d in self.calibration_data)
        min_hand_y = min(d['hand_y'] for d in self.calibration_data)
        max_hand_y = max(d['hand_y'] for d in self.calibration_data)

        min_screen_x = min(d['screen_x'] for d in self.calibration_data)
        max_screen_x = max(d['screen_x'] for d in self.calibration_data)
        min_screen_y = min(d['screen_y'] for d in self.calibration_data)
        max_screen_y = max(d['screen_y'] for d in self.calibration_data)

        hand_x_range = max_hand_x - min_hand_x
        hand_y_range = max_hand_y - min_hand_y

        if hand_x_range == 0: hand_x_range = 0.001
        if hand_y_range == 0: hand_y_range = 0.001

        clamped_hand_x = np.clip(hand_x, min_hand_x, max_hand_x)
        clamped_hand_y = np.clip(hand_y, min_hand_y, max_hand_y)

        normalized_hand_x = (clamped_hand_x - min_hand_x) / hand_x_range
        normalized_hand_y = (clamped_hand_y - min_hand_y) / hand_y_range

        # FIX 1: Flip X axis — camera already mirrors so we invert the X mapping
        screen_x = min_screen_x + (1.0 - normalized_hand_x) * (max_screen_x - min_screen_x)
        screen_y = min_screen_y + normalized_hand_y * (max_screen_y - min_screen_y)

        screen_x = np.clip(screen_x, 0, SCREEN_WIDTH)
        screen_y = np.clip(screen_y, 0, SCREEN_HEIGHT)

        return int(screen_x), int(screen_y)

    def run(self):
        while self.running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        self.running = False

            delta_time = self.clock.tick(60) / 1000.0

            # Read a frame from the webcam (no flip — camera already mirrors)
            ret, frame = self.cap.read()
            if not ret:
                print("Error: Failed to read frame from camera.")
                self.running = False
                continue

            frame_rgb_mp = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb_mp)
            detection_result = self.landmarker.detect(mp_image)

            self.hand_landmarks = None
            if detection_result.hand_landmarks:
                self.hand_landmarks = detection_result.hand_landmarks[0]

            annotated_frame = frame_rgb_mp.copy()
            if detection_result.hand_landmarks:
                h, w = annotated_frame.shape[:2]
                HAND_CONNECTIONS = [
                    (0,1),(1,2),(2,3),(3,4),
                    (0,5),(5,6),(6,7),(7,8),
                    (5,9),(9,10),(10,11),(11,12),
                    (9,13),(13,14),(14,15),(15,16),
                    (13,17),(17,18),(18,19),(19,20),(0,17)
                ]
                for hand_lms in detection_result.hand_landmarks:
                    pts = [(int(lm.x * w), int(lm.y * h)) for lm in hand_lms]
                    for start, end in HAND_CONNECTIONS:
                        cv2.line(annotated_frame, pts[start], pts[end], (0, 255, 0), 2)
                    for pt in pts:
                        cv2.circle(annotated_frame, pt, 4, (255, 0, 0), -1)

            frame_rgb_pygame = annotated_frame
            self.camera_surface = pygame.surfarray.make_surface(np.rot90(frame_rgb_pygame)).convert()
            self.camera_surface = pygame.transform.scale(self.camera_surface, (SCREEN_WIDTH, SCREEN_HEIGHT))

            # Game Logic based on state
            if self.game_state == GAME_STATE_CALIBRATION:
                if self.calibration_cooldown_timer > 0:
                    self.calibration_cooldown_timer -= delta_time
                else:
                    if self.hand_landmarks and self._is_finger_gun(self.hand_landmarks):
                        self.calibration_timer += delta_time
                        if self.calibration_timer >= CALIBRATION_HOLD_TIME:
                            current_target_pos = CALIBRATION_TARGET_POSITIONS[self.current_calibration_target_index]
                            index_tip = self.hand_landmarks[self.mp_hands_landmark.INDEX_FINGER_TIP]
                            self.calibration_data.append({
                                'screen_x': current_target_pos[0],
                                'screen_y': current_target_pos[1],
                                'hand_x': index_tip.x,
                                'hand_y': index_tip.y
                            })
                            print(f"Calibrated point {self.current_calibration_target_index + 1}: Screen({current_target_pos[0]:.0f}, {current_target_pos[1]:.0f}), Hand({index_tip.x:.2f}, {index_tip.y:.2f})")

                            self.calibration_timer = 0.0
                            self.current_calibration_target_index += 1
                            self.calibration_cooldown_timer = CALIBRATION_POINT_COOLDOWN

                            if self.current_calibration_target_index >= len(CALIBRATION_TARGET_POSITIONS):
                                self.game_state = GAME_STATE_PLAYING
                                print("Calibration complete! Starting game...")
                    else:
                        self.calibration_timer = 0.0

            elif self.game_state == GAME_STATE_PLAYING:
                if self.hand_landmarks and self.calibration_data:
                    index_tip = self.hand_landmarks[self.mp_hands_landmark.INDEX_FINGER_TIP]
                    self.cursor_x, self.cursor_y = self._map_hand_to_screen(index_tip.x, index_tip.y)

                    if self._is_finger_gun(self.hand_landmarks) and self._is_thumb_lowered(self.hand_landmarks):
                        current_time = pygame.time.get_ticks() / 1000.0
                        if current_time - self.last_shot_time > self.SHOT_COOLDOWN:
                            print(f"SHOOT! at ({self.cursor_x}, {self.cursor_y})")
                            self.last_shot_time = current_time

                            bullet_rect = pygame.Rect(self.cursor_x - 5, self.cursor_y - 5, 10, 10)
                            for duck in list(self.ducks):
                                if bullet_rect.colliderect(duck.rect):
                                    duck.kill()
                                    self.score += 1
                                    print(f"Duck hit! Score: {self.score}")

                    self.duck_spawn_timer += delta_time
                    if self.duck_spawn_timer >= DUCK_SPAWN_INTERVAL:
                        self._spawn_duck()
                        self.duck_spawn_timer = 0.0

                    self.ducks.update()

                    for duck in list(self.ducks):
                        if duck.rect.x < -duck.rect.width or duck.rect.x > SCREEN_WIDTH + duck.rect.width:
                            duck.kill()

            # --- Drawing ---
            self.screen.fill((0, 0, 0))
            self.screen.blit(self.background_image, (0, 0))

            if self.camera_surface:
                self.screen.blit(self.camera_surface, (0, 0))

            if self.game_state == GAME_STATE_CALIBRATION:
                if self.current_calibration_target_index < len(CALIBRATION_TARGET_POSITIONS):
                    target_x, target_y = CALIBRATION_TARGET_POSITIONS[self.current_calibration_target_index]
                    # FIX 2: No Y inversion — coords are already in Pygame space (top-left origin)
                    target_rect = self.target_image.get_rect(center=(int(target_x), int(target_y)))
                    self.screen.blit(self.target_image, target_rect)

                    font = pygame.font.Font(None, 30)
                    text_surface = font.render(
                        f"Calibration: Point at target {self.current_calibration_target_index + 1}/{len(CALIBRATION_TARGET_POSITIONS)}",
                        True, (255, 255, 255)
                    )
                    self.screen.blit(text_surface, (10, SCREEN_HEIGHT - 30 - text_surface.get_height()))

                    text_surface = font.render(
                        f"Hold finger gun: {self.calibration_timer:.1f}/{CALIBRATION_HOLD_TIME:.1f}s",
                        True, (255, 255, 255)
                    )
                    self.screen.blit(text_surface, (10, SCREEN_HEIGHT - 60 - text_surface.get_height()))

                    if self.calibration_cooldown_timer > 0:
                        text_surface = font.render(
                            f"Next target in: {self.calibration_cooldown_timer:.1f}s",
                            True, (255, 255, 0)
                        )
                        self.screen.blit(text_surface, (10, SCREEN_HEIGHT - 90 - text_surface.get_height()))
                else:
                    font = pygame.font.Font(None, 40)
                    text_surface = font.render(
                        "Calibration Complete! Starting Game...",
                        True, (255, 255, 255)
                    )
                    text_rect = text_surface.get_rect(center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2))
                    self.screen.blit(text_surface, text_rect)

            elif self.game_state == GAME_STATE_PLAYING:
                # FIX 3: No Y inversion on cursor — cursor_y is already in Pygame coords
                pygame.draw.circle(self.screen, (255, 0, 0), (self.cursor_x, self.cursor_y), 15)
                self.ducks.draw(self.screen)

                font = pygame.font.Font(None, 30)
                score_text_surface = font.render(f"Score: {self.score}", True, (255, 255, 255))
                self.screen.blit(score_text_surface, (SCREEN_WIDTH - 150, SCREEN_HEIGHT - 30 - score_text_surface.get_height()))

                game_on_text_surface = font.render("GAME ON!", True, (255, 255, 255))
                self.screen.blit(game_on_text_surface, (10, SCREEN_HEIGHT - 30 - game_on_text_surface.get_height()))

            if DEBUG_GESTURES:
                self._draw_debug_overlay()

            pygame.display.flip()

        self.cap.release()
        pygame.quit()


class Duck(pygame.sprite.Sprite):
    def __init__(self, image, scale):
        super().__init__()
        self.image = pygame.transform.scale_by(image, scale)
        self.rect = self.image.get_rect()
        self.change_x = 0

    def update(self):
        self.rect.x += self.change_x


if __name__ == "__main__":
    game = Game()
    game.run()
