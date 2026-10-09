"""Local port of JARVIS gesture classification and desktop interaction semantics.

Camera frames stay in Astra's controller process. This module accepts only 21
MediaPipe hand landmarks and emits local cursor/window actions.
"""
from __future__ import annotations

import collections
import ctypes
import math
import os
import time

# JARVIS gesture thresholds and timing.
FINGER_ON_ANG, FINGER_OFF_ANG = 55.0, 70.0
FINGER_ON_FAR, FINGER_OFF_FAR = 1.08, 0.98
THUMB_ON, THUMB_OFF = 1.20, 1.05
FIST_THUMB_MAX = 1.0
POSE_STABLE_FRAMES = 2
PINCH_ON, PINCH_OFF = 0.26, 0.38
PINCH_PALM_ON, PINCH_PALM_KEEP = 0.52, 0.45
PINCH_INDEX_ANG_ON, PINCH_INDEX_ANG_KEEP = 125.0, 138.0
RIGHT_PINCH_ON, RIGHT_PINCH_OFF = 0.24, 0.36
RIGHT_INDEX_CLEAR = 0.45
RIGHT_PALM_ON, RIGHT_MID_ANG_MAX, RIGHT_MID_FAR_MIN = 0.56, 115.0, 0.82
RIGHT_CLICK_MAX_S = 0.9
HOLD_ARM_S, CLICK_MIN_S, CLICK_MIN_FRAMES = 0.34, 0.05, 2
PINCH_RELEASE_S, HAND_LOST_GRACE_S = 0.045, 0.25
DOUBLE_CLICK_SNAP_PX = 36
CURSOR_MIN_CUTOFF, CURSOR_BETA, CURSOR_D_CUTOFF = 0.9, 8.0, 1.5
CURSOR_DEADBAND_PX, CURSOR_BOX_K = 1.5, 3.6
CURSOR_BOX_MIN, CURSOR_BOX_MAX, CURSOR_BOX_CY = 0.30, 0.72, 0.47
HAND_SIZE_TAU_S, OFFSET_DECAY_TAU_S = 1.2, 0.18
SCROLL_DEAD, SCROLL_FULL, SCROLL_MAX_NOTCHES_S, SCROLL_SETTLE_S = 0.28, 1.30, 20.0, 0.12
VICTORY_CMD_HOLD_S = 0.9
WIN_ENGAGE_SPREAD, WIN_KEEP_SPREAD, WIN_ENGAGE_S = 0.22, 0.01, 0.30
WIN_UPDATE_GAP_S, WIN_MAX_SPREAD, WIN_MOVE_GAIN = 0.032, 0.92, 3.4
WIN_MOVE_DEAD, WIN_MOVE_SOFT, WIN_HOLD_SPREAD = 0.0045, 0.010, 0.28
STAGE_CARD, STAGE_SIDE, STAGE_MAIN = (0.30, 0.16, 0.40, 0.68), (0.00, 0.00, 0.30, 1.00), (0.28, 0.00, 0.72, 1.00)
CMD_HOLD_S, CMD_COOLDOWN_S, FIST_MINIMIZE_S, MINIMIZE_COOLDOWN_S = 0.5, 1.6, 0.55, 1.2
COMMAND_POSES = frozenset({"victory", "three", "horns", "shaka", "ily", "four"})


def dist(a, b):
    return math.sqrt(sum((a[i] - b[i]) ** 2 for i in range(len(a))))


def angle(a, b, c):
    v1 = [b[i] - a[i] for i in range(len(a))]
    v2 = [c[i] - b[i] for i in range(len(a))]
    n1, n2 = math.sqrt(sum(x*x for x in v1)), math.sqrt(sum(x*x for x in v2))
    if n1 < 1e-9 or n2 < 1e-9:
        return 0.0
    return math.degrees(math.acos(max(-1.0, min(1.0, sum(v1[i]*v2[i] for i in range(len(a))) / (n1*n2)))))


def mean(points):
    return tuple(sum(p[i] for p in points) / len(points) for i in range(len(points[0])))


def classify_fingers(fingers):
    t, i, m, r, p = fingers
    n = i + m + r + p
    if n == 0:
        return "thumb" if t else "fist"
    if n == 4:
        return "palm" if t else "four"
    if i and not m and not r and not p:
        return "point"
    if i and m and not r and not p:
        return "victory"
    if i and m and r and not p:
        return "three"
    if i and p and not m and not r:
        return "ily" if t else "horns"
    if p and not i and not m and not r and t:
        return "shaka"
    return None


def window_engage(fingers, spread):
    return (sum(fingers[1:]) == 4 or (sum(fingers[1:]) == 3 and fingers[0])) and spread >= WIN_ENGAGE_SPREAD


def window_keep(fingers, spread):
    t, i, m, r, p = fingers
    other, total = i + m + r + p, t + i + m + r + p
    if total == 0 or (other == 1 and i and not t) or (other == 2 and i and m and not t):
        return False
    return other >= 2 or spread >= WIN_KEEP_SPREAD or (t + p) >= 1


class FeatureTracker:
    """JARVIS hand feature extraction with hysteretic 3D finger-state detection."""
    def __init__(self):
        self.fingers = [0] * 5

    def reset(self):
        self.fingers = [0] * 5

    def update(self, landmarks, world, aspect):
        p = [(1.0 - float(q.x), float(q.y)) for q in landmarks]
        w = [(float(q.x), float(q.y), float(q.z)) for q in world] if world and len(world) == 21 else None
        g = w if w else p
        palm_span = dist(g[0], g[9]) or 1e-6
        palm_width = dist(g[5], g[17]) or 1e-6
        thumb_ratio = dist(g[4], g[13]) / palm_width
        fingers = [int(thumb_ratio > (THUMB_OFF if self.fingers[0] else THUMB_ON))]
        angles, fars = [], []
        for k, mcp in enumerate((5, 9, 13, 17), start=1):
            a = angle(g[mcp], g[mcp+1], g[mcp+3])
            far = dist(g[mcp+3], g[0]) / (dist(g[mcp+1], g[0]) or 1e-6)
            angles.append(a); fars.append(far)
            fingers.append(int(a < (FINGER_OFF_ANG if self.fingers[k] else FINGER_ON_ANG)
                                and far > (FINGER_OFF_FAR if self.fingers[k] else FINGER_ON_FAR)))
        self.fingers = fingers
        span2 = dist(p[0], p[9]) or 1e-6
        pinch, mid_pinch = dist(p[4], p[8]) / span2, dist(p[4], p[12]) / span2
        palm_c = mean([g[i] for i in (0, 5, 9, 13, 17)])
        pinch_palm = dist(mean([g[4], g[8]]), palm_c) / palm_span
        mid_palm = dist(mean([g[4], g[12]]), palm_c) / palm_span
        spread_ratio = dist(g[4], g[20]) / palm_width
        spread = max(0.0, min(1.0, (spread_ratio - 1.05) / 2.15))
        anchor = (p[5][0] * .7 + p[9][0] * .3, p[5][1] * .7 + p[9][1] * .3)
        palm = ((p[5][0] + p[17][0]) * .5, (p[5][1] + p[17][1]) * .5)
        return {"fingers": fingers, "pinch": pinch, "pinch_palm": pinch_palm,
                "index_ang": angles[0], "mid_pinch": mid_pinch, "mid_palm": mid_palm,
                "mid_ang": angles[1], "mid_far": fars[1], "spread": spread,
                "anchor": anchor, "palm": palm,
                "size": max(span2, dist(p[5], p[17]) * 1.25), "aspect": aspect,
                "thumb_ratio": thumb_ratio}


class OneEuro2D:
    def __init__(self):
        self.reset()
    def reset(self):
        self.x = self.t = None; self.dx = (0., 0.)
    @staticmethod
    def alpha(cutoff, dt):
        return 1.0 / (1.0 + 1.0 / (2.0 * math.pi * cutoff) / dt)
    def __call__(self, x, t, scale):
        if self.x is None:
            self.x, self.t = x, t; return x
        dt = max(1e-3, min(.5, t-self.t)); self.t = t
        raw = ((x[0]-self.x[0])/dt, (x[1]-self.x[1])/dt)
        a = self.alpha(CURSOR_D_CUTOFF, dt)
        self.dx = (a*raw[0]+(1-a)*self.dx[0], a*raw[1]+(1-a)*self.dx[1])
        speed = math.hypot(*self.dx) / max(1e-6, scale)
        a = self.alpha(CURSOR_MIN_CUTOFF + CURSOR_BETA*speed, dt)
        self.x = (a*x[0]+(1-a)*self.x[0], a*x[1]+(1-a)*self.x[1])
        return self.x


class _Windows:
    """Small local Win32 adapter for Jarvis's deliberate palm/fist window gestures."""
    def __init__(self, pyautogui):
        self.pg = pyautogui
        self.u = ctypes.windll.user32 if os.name == "nt" else None
        self.armed = None
    def screen(self):
        w, h = self.pg.size(); return (0, 0, int(w), int(h))
    def foreground(self):
        try:
            h = self.u.GetForegroundWindow()
            return h if h and self.u.IsWindowVisible(h) else None
        except Exception: return None
    def maximize(self, hwnd):
        try: self.u.ShowWindowAsync(hwnd, 3)
        except Exception: pass
    def minimize(self, hwnd):
        try: self.u.ShowWindowAsync(hwnd, 6); return True
        except Exception: return False
    def rect(self, hwnd):
        class R(ctypes.Structure):
            _fields_=[("l",ctypes.c_long),("t",ctypes.c_long),("r",ctypes.c_long),("b",ctypes.c_long)]
        r=R()
        try:
            if self.u.GetWindowRect(hwnd,ctypes.byref(r)): return (r.l,r.t,r.r-r.l,r.b-r.t)
        except Exception: pass
        return None
    def set_rect(self, hwnd, x, y, w, h):
        try: self.u.SetWindowPos(hwnd,None,int(x),int(y),int(w),int(h),0x0010|0x0040|0x0004)
        except Exception: pass
    def stage(self, active):
        x,y,sw,sh=self.screen()
        rects=[]
        for fx,fy,fw,fh in [STAGE_CARD, STAGE_SIDE, STAGE_MAIN]:
            rects.append((x+round(sw*fx),y+round(sh*fy),max(240,round(sw*fw)),max(160,round(sh*fh))))
        if active and self.rect(active): self.set_rect(active,*rects[0])
        return {"hwnd":active,"work":(x,y,x+sw,y+sh),"card":rects[0],"cx":rects[0][0]+rects[0][2]/2,
                "cy":rects[0][1]+rects[0][3]/2,"px":None,"py":None,"max":False,"t_set":time.monotonic()}
    def update_palm(self, s, spread, nx, ny, now):
        if not s or not s["hwnd"]: return "palm"
        hwnd=s["hwnd"]
        if spread >= WIN_MAX_SPREAD:
            if not s["max"]: self.maximize(hwnd); s["max"]=True
            s["px"],s["py"]=nx,ny; return "fullscreen"
        if s["max"]:
            try: self.u.ShowWindowAsync(hwnd,9)
            except Exception: pass
            s["max"]=False
        x,y,sw,sh=s["work"]
        card=s["card"]; ww,hh=card[2],card[3]
        if spread < WIN_HOLD_SPREAD:
            frac=max(.42,spread/max(WIN_HOLD_SPREAD,1e-3)); ww=max(240,int(card[2]*frac)); hh=max(160,int(card[3]*frac))
        if s["px"] is None: s["px"],s["py"]=nx,ny
        dx,dy=nx-s["px"],ny-s["py"]; s["px"],s["py"]=nx,ny
        mag=math.hypot(dx,dy)
        if mag<WIN_MOVE_DEAD: dx=dy=0.
        elif mag<WIN_MOVE_SOFT:
            k=((mag-WIN_MOVE_DEAD)/(WIN_MOVE_SOFT-WIN_MOVE_DEAD))**2; dx*=k; dy*=k
        s["cx"]+=dx*sw*WIN_MOVE_GAIN; s["cy"]+=dy*sh*WIN_MOVE_GAIN
        left=max(x,min(x+sw-ww,int(s["cx"]-ww/2))); top=max(y,min(y+sh-hh,int(s["cy"]-hh/2)))
        s["cx"],s["cy"]=left+ww/2,top+hh/2
        if now-s["t_set"]>=WIN_UPDATE_GAP_S:
            self.set_rect(hwnd,left,top,ww,hh); s["t_set"]=now
        return "window"


class JarvisGestureEngine:
    """Astra-adapted JARVIS gesture state machine; the camera is started by Astra UI."""
    def __init__(self, pyautogui, gesture_settings=None):
        self.pg=pyautogui; self.win=_Windows(pyautogui); self.filter=OneEuro2D()
        self.gesture_settings = gesture_settings or {"enabled": {"cursor": True, "clicks": True, "scroll": True, "window": True, "minimize": True}, "custom": {}}
        self.custom_fired = None
        self.reset()
    def reset(self):
        self.pose=self.pose_cand=None; self.pose_n=0; self.hand_size=None; self.hand_size_t=None
        self.hist=collections.deque(maxlen=24); self.cursor_active=False; self.cursor_last=None
        self.left={"state":"up"}; self.right={"state":"up"}; self.last_click=-1e9; self.last_click_pos=None
        self.last_hand=-1e9; self.scroll=None; self.engage_since=None; self.window_state=None
        self.fist_since=None; self.fist_fired=False; self.minimize_until=0.; self.cmd_pose=None; self.cmd_since=0.; self.cmd_fired=False
        self.last_label="waiting_for_hand"; self.offset=(0.,0.); self.offset_t=0.; self.custom_fired=None; self.filter.reset()
        try: self.pg.mouseUp()
        except Exception: pass
    def _stable(self, raw):
        if raw==self.pose_cand: self.pose_n+=1
        else: self.pose_cand=raw; self.pose_n=1
        if self.pose_n>=POSE_STABLE_FRAMES: self.pose=raw
        return self.pose
    def _size(self,f,now,force=False):
        if self.hand_size is None: self.hand_size=f["size"]
        elif force:
            dt=max(0.,now-(self.hand_size_t or now)); a=1-math.exp(-dt/HAND_SIZE_TAU_S); self.hand_size+=(f["size"]-self.hand_size)*a
        self.hand_size_t=now; return self.hand_size
    def _target(self,f):
        sw,sh=self.pg.size(); size=self.hand_size or f["size"]
        bw=max(CURSOR_BOX_MIN,min(CURSOR_BOX_MAX,size*CURSOR_BOX_K)); bh=min(.9,bw*f["aspect"]*.5625)
        u=max(0.,min(1.,(f["anchor"][0]-(.5-bw*.5))/bw)); v=max(0.,min(1.,(f["anchor"][1]-(CURSOR_BOX_CY-bh*.5))/bh))
        return u*(sw-1),v*(sh-1)
    def _move(self,f,now):
        sw,sh=self.pg.size(); x,y=self.filter(self._target(f),now,float(max(sw,sh)))
        self.hist.append((now,x,y)); ox,oy=self.offset
        px=max(0,min(sw-1,int(round(x+ox)))); py=max(0,min(sh-1,int(round(y+oy))))
        if self.cursor_last is None or math.hypot(px-self.cursor_last[0],py-self.cursor_last[1])>=CURSOR_DEADBAND_PX:
            self.pg.moveTo(max(2, min(sw - 3, px)), max(2, min(sh - 3, py)), _pause=False); self.cursor_last=(px,py)
        self.cursor_active=True
    def _release(self):
        self.cursor_active=False; self.filter.reset(); self.hist.clear(); self.offset=(0.,0.)
    def _left_on(self,f):
        if self.left["state"]=="up": return f["pinch"]<PINCH_ON and f["pinch_palm"]>PINCH_PALM_ON and f["index_ang"]<PINCH_INDEX_ANG_ON and f["pinch"]<=f["mid_pinch"]
        return f["pinch"]<PINCH_OFF and f["pinch_palm"]>PINCH_PALM_KEEP and f["index_ang"]<PINCH_INDEX_ANG_KEEP
    def _right_on(self,f):
        if self.right["state"]=="up": return f["mid_pinch"]<RIGHT_PINCH_ON and f["pinch"]>RIGHT_INDEX_CLEAR and f["mid_palm"]>RIGHT_PALM_ON and f["mid_ang"]<RIGHT_MID_ANG_MAX and f["mid_far"]>RIGHT_MID_FAR_MIN
        return f["mid_pinch"]<RIGHT_PINCH_OFF and f["pinch"]>RIGHT_INDEX_CLEAR*.8
    def _press(self, state, now):
        pos=self.pg.position(); pos=(int(pos.x),int(pos.y))
        if state is self.left and self.last_click_pos is not None and now-self.last_click<=.5 and math.dist(pos,self.last_click_pos)<=DOUBLE_CLICK_SNAP_PX: pos=self.last_click_pos
        state.update(state="down",t_down=now,t_on=now,frames=1,t_off=None,freeze=pos)
        self.pg.moveTo(*pos,_pause=False)
    def _finish_left(self,now,cancel=False):
        s=self.left
        if s["state"]=="drag":
            self.pg.mouseUp(button="left"); self.last_label="point"
        elif s["state"]=="down" and not cancel:
            dur=s["t_on"]-s["t_down"]
            if s["frames"]>=CLICK_MIN_FRAMES and dur>=CLICK_MIN_S-1e-3:
                self.pg.moveTo(*s["freeze"],_pause=False); self.pg.click(button="left"); self.last_click=now; self.last_click_pos=s["freeze"]; self.last_label="click"
        self.left={"state":"up"}
    def _finish_right(self,now,cancel=False):
        s=self.right
        if s["state"]=="down" and not cancel and s["frames"]>=CLICK_MIN_FRAMES and CLICK_MIN_S-1e-3<=s["t_on"]-s["t_down"]<=RIGHT_CLICK_MAX_S:
            self.pg.moveTo(*s["freeze"],_pause=False); self.pg.click(button="right"); self.last_label="right_click"
        self.right={"state":"up"}
    def _pinches(self,f,now):
        left=self._left_on(f) if self.right["state"]=="up" else False; s=self.left
        if left:
            if s["state"]=="up": self._press(s,now); self.last_label="pinch"
            else:
                s["t_off"]=None; s["t_on"]=now; s["frames"]+=1
                if s["state"]=="down" and now-s["t_down"]>=HOLD_ARM_S:
                    s["state"]="drag"; self.pg.moveTo(*s["freeze"],_pause=False); self.pg.mouseDown(button="left"); self.filter.reset(); self.hist.clear(); self.last_label="drag"
            if s["state"]=="drag": self._move(f,now)
            return True
        if s["state"]!="up":
            if s.get("t_off") is None: s["t_off"]=now
            if now-s["t_off"]>=PINCH_RELEASE_S: self._finish_left(now); return False
            if s["state"]=="drag": self._move(f,now)
            return True
        right=self._right_on(f); s=self.right
        if right:
            if s["state"]=="up": self._press(s,now); self.last_label="pinch"
            else: s["t_off"]=None; s["t_on"]=now; s["frames"]+=1
            return True
        if s["state"]!="up":
            if s.get("t_off") is None: s["t_off"]=now
            if now-s["t_off"]>=PINCH_RELEASE_S: self._finish_right(now); return False
            return True
        return False
    def no_hand(self,now):
        if now-self.last_hand<HAND_LOST_GRACE_S: return
        if self.left["state"]!="up": self._finish_left(now,True)
        if self.right["state"]!="up": self._finish_right(now,True)
        self._release(); self.pose=self.pose_cand=None; self.pose_n=0; self.engage_since=None; self.scroll=None; self.window_state=None; self.fist_since=None; self.fist_fired=False; self.last_label="waiting_for_hand"
    def update(self,f,now):
        self.last_hand=now; pose=self._stable(classify_fingers(f["fingers"]))
        if pose != self.custom_fired:
            self.custom_fired=None
        custom_action = self.gesture_settings.get("custom", {}).get(pose)
        custom_enabled = self.gesture_settings.get("custom_enabled", {}).get(pose, True)
        if custom_action and custom_enabled:
            if self.custom_fired != pose:
                self._run_custom_action(custom_action)
                self.custom_fired=pose
            self._release(); self.last_label=f"{pose} → {custom_action}"; return
        if custom_action and not custom_enabled:
            self._release(); self.last_label=pose; return
        if not self.gesture_settings.get("enabled", {}).get("clicks", True) and (self.left["state"] != "up" or self.right["state"] != "up"):
            try: self.pg.mouseUp()
            except Exception: pass
            self.left={"state":"up"}; self.right={"state":"up"}
        if not self.gesture_settings.get("enabled", {}).get("window", True):
            self.window_state=None; self.engage_since=None
        if self.window_state:
            if window_keep(f["fingers"],f["spread"]):
                self.last_label=self.win.update_palm(self.window_state,f["spread"],*f["palm"],now); return
            self.window_state=None; self.engage_since=None
        if self.gesture_settings.get("enabled", {}).get("clicks", True) and self._pinches(f,now): self.engage_since=None; self.scroll=None; self.fist_since=None; return
        if self.gesture_settings.get("enabled", {}).get("window", True) and window_engage(f["fingers"],f["spread"]):
            self.scroll=None; self.fist_since=None
            if self.engage_since is None: self.engage_since=now
            elif now-self.engage_since>=WIN_ENGAGE_S:
                hwnd=self.win.foreground()
                if hwnd:
                    self._release(); self.window_state=self.win.stage(hwnd); self.last_label=self.win.update_palm(self.window_state,f["spread"],*f["palm"],now)
                else: self.engage_since=now
            return
        self.engage_since=None
        if pose!="victory" or not self.gesture_settings.get("enabled", {}).get("scroll", True): self.scroll=None
        if pose!="fist" or not self.gesture_settings.get("enabled", {}).get("minimize", True): self.fist_since=None; self.fist_fired=False
        if pose not in COMMAND_POSES: self.cmd_pose=None
        if pose=="point" and self.gesture_settings.get("enabled", {}).get("cursor", True): self._size(f,now,True); self._move(f,now); self.last_label="point"; return
        if pose=="point" and not self.gesture_settings.get("enabled", {}).get("cursor", True):
            self._release(); self.last_label="point"; return
        self._release()
        if pose == "victory":
            if self.gesture_settings.get("enabled", {}).get("scroll", True):
                self._victory(f,now)
            else:
                self.last_label="victory"
        elif pose in COMMAND_POSES: self._command_pose(pose,now,VICTORY_CMD_HOLD_S if pose=="victory" else CMD_HOLD_S)
        elif pose == "fist":
            if self.gesture_settings.get("enabled", {}).get("minimize", True):
                self._fist(f,now)
            else:
                self.last_label="fist"
        elif pose: self.last_label=pose
    def _run_custom_action(self, action):
        try:
            if action == "left_click": self.pg.click(button="left")
            elif action == "right_click": self.pg.click(button="right")
            elif action == "double_click": self.pg.doubleClick(button="left")
            elif action == "scroll_up": self.pg.scroll(3 * 120)
            elif action == "scroll_down": self.pg.scroll(-3 * 120)
            elif action == "minimize_window":
                hwnd=self.win.foreground()
                if hwnd: self.win.minimize(hwnd)
        except Exception:
            pass

    def _command_pose(self,pose,now,hold):
        if self.cmd_pose!=pose: self.cmd_pose=pose; self.cmd_since=now; self.cmd_fired=False
        # Jarvis constructor bindings are private IPC actions; Astra deliberately does not execute them.
        self.last_label=pose
    def _victory(self,f,now):
        size=self._size(f,now); y=f["anchor"][1]; s=self.scroll
        if s is None: s=self.scroll={"y0":y,"y":y,"t0":now,"t":now,"acc":0.,"moved":False}; self.cmd_pose=None
        dt=max(0.,min(.2,now-s["t"])); s["t"]=now; s["y"]+=(y-s["y"])*.5
        if now-s["t0"]<SCROLL_SETTLE_S: s["y0"]=s["y"]; self.last_label="victory"; return
        off=(s["y0"]-s["y"])/max(size,1e-3); mag=abs(off)
        if mag<=SCROLL_DEAD:
            s["acc"]=0.; self._command_pose("victory",now,VICTORY_CMD_HOLD_S) if not s["moved"] else None; return
        s["moved"]=True; self.cmd_pose=None; k=min(1.,(mag-SCROLL_DEAD)/(SCROLL_FULL-SCROLL_DEAD)); speed=SCROLL_MAX_NOTCHES_S*k**1.5; sign=1. if off>0 else -1.
        if s["acc"]==0. or (s["acc"]>0)!=(sign>0): s["acc"]=sign*.7
        s["acc"]+=sign*speed*dt; n=int(s["acc"])
        if n: s["acc"]-=n; self.pg.scroll(int(n*120)); self.last_label="scroll_up" if n>0 else "scroll_down"
    def _fist(self,f,now):
        if self.fist_fired: return
        if f["thumb_ratio"]>FIST_THUMB_MAX: self.fist_since=None; return
        if self.fist_since is None: self.fist_since=now; return
        if now-self.fist_since>=FIST_MINIMIZE_S and now>=self.minimize_until:
            self.fist_fired=True; hwnd=self.win.foreground()
            if hwnd and self.win.minimize(hwnd): self.minimize_until=now+MINIMIZE_COOLDOWN_S; self.last_label="minimize"
    def shutdown(self):
        try:
            self.pg.mouseUp()
        except Exception: pass
