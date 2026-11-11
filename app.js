// 撮影判定テスト — 勉強している時間だけを数えられるかを試す。
// 映像は端末の中で MediaPipe に通すだけで、保存も送信もしない。

const APP_VER = 'v1';
const MP_VER = '1.2.2';
const MP_CDN = `https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@${MP_VER}`;
const M = 'https://storage.googleapis.com/mediapipe-models';
const MODEL = {
  pose: `${M}/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task`,
  face: `${M}/face_landmarker/face_landmarker/float16/1/face_landmarker.task`,
  objGPU: `${M}/object_detector/efficientdet_lite0/float16/1/efficientdet_lite0.tflite`,
  objCPU: `${M}/object_detector/efficientdet_lite0/int8/1/efficientdet_lite0.tflite`,
};

const qs = new URLSearchParams(location.search);
const SPEED = Math.max(1, Number(qs.get('speed')) || 1); // 試験の時間を縮める（PCでの確認用）
const FORCE_CPU = qs.has('cpu');

const DEFAULTS = { jit: 0.010, diff: 1.5, blink: 20, hold: 5, obj: 0.35 };
const WIN_MS = 10000;      // 揺れ・画素の変化を見る幅
const PROC_MS = 66;        // 処理の間隔（約15fps）
const OBJ_MS = 1000;       // 物体検出の間隔
const HOLD_LABELS = ['cell phone', 'remote'];
const HAND_POINTS = [15, 16, 17, 18, 19, 20, 21, 22]; // 手首・小指・人差し指・親指
const STATE_JA = { count: '数えている', away: '止まっている：人がいない', still: '止まっている：動きがない', phone: '止まっている：スマホを持っている' };
const STATE_LETTER = { count: 'C', away: 'A', still: 'S', phone: 'P' };
const LABEL_JA = {
  'person': '人', 'cell phone': 'スマホ', 'remote': 'リモコン', 'book': '本', 'laptop': 'ノートPC',
  'dining table': '机', 'tv': '画面', 'keyboard': 'キーボード', 'mouse': 'マウス', 'cup': 'コップ',
  'bottle': 'ボトル', 'chair': '椅子', 'clock': '時計', 'scissors': 'はさみ', 'teddy bear': 'ぬいぐるみ',
};

const SCENES = [
  { id: 'write', name: '書いて勉強', sec: 120, prep: 10, expect: 'count',
    text: 'ノートに書く・計算するなど、普段どおりに勉強してください。' },
  { id: 'read', name: '読んで勉強', sec: 120, prep: 10, expect: 'count',
    text: '本や紙を読むだけにして、なるべく手を動かさず、じっと読んでください。' },
  { id: 'phone', name: 'スマホを持つ', sec: 90, prep: 10, expect: 'phone',
    text: '別のスマホ（無ければタブレット）を手に持って、普段どおりに触ってください。' },
  { id: 'game', name: 'ゲーム機を持つ', sec: 60, prep: 10, expect: 'phone',
    text: 'ゲーム機を手に持って遊ぶ姿勢をしてください。持っていなければ「この場面を飛ばす」。' },
  { id: 'away', name: '席を外す', sec: 60, prep: 15, expect: 'away',
    text: '始めたらカメラに映らない所へ離れ、1分半ほどたってから戻ってください。' },
  { id: 'photo', name: '写真を立てる', sec: 90, prep: 20, expect: 'still',
    text: '人の写真（紙、または別の画面に出した写真）をカメラの前に立て、自分は映らない所へ離れてください。2分ほどたったら戻ってください。' },
];

const $ = (id) => document.getElementById(id);
const video = $('video');
const overlay = $('overlay');
const octx = overlay.getContext('2d');
const small = document.createElement('canvas');
const sctx = small.getContext('2d', { willReadFrequently: true });

let P = { ...DEFAULTS };
try { Object.assign(P, JSON.parse(localStorage.getItem('studycam.params') || '{}')); } catch (_) {}

const S = {
  appVer: APP_VER, delegate: null, loadMs: 0, started: false,
  t0: 0, lastProc: 0, lastFrameT: 0, fpsTimes: [],
  poseWin: [], diffWin: [], snaps: [], lastSnapT: -Infinity,
  blinkClosed: false, lastBlink: -Infinity, blinks: 0,
  det: { t: -Infinity, list: [] }, lastObjT: -Infinity, deskT: -Infinity,
  raw: null, rawHist: [], disp: null,
  totals: { count: 0, away: 0, still: 0, phone: 0 },
  log: [], lastLogT: -Infinity,
  scene: null, sceneStats: {}, results: [],
  cur: {}, errors: [], loopErrors: 0,
};
window.__studycam = { S, P, SCENES };

let vision = null, pose = null, face = null, obj = null;

// ---------- 読み込み ----------

function showError(msg) {
  S.errors.push(String(msg));
  const el = $('error');
  el.hidden = false;
  el.textContent = S.errors.join('\n');
}

function setLoading(msg) {
  const el = $('loading');
  if (msg == null) { el.hidden = true; return; }
  el.hidden = false;
  $('loadMsg').textContent = msg;
}

async function createModels(delegate) {
  const mp = await import(`${MP_CDN}/vision_bundle.mjs`);
  if (!vision) vision = await mp.FilesetResolver.forVisionTasks(`${MP_CDN}/wasm`);
  setLoading(`判定用のAIを読み込み中（1/3 体の位置・${delegate}）…`);
  pose = await mp.PoseLandmarker.createFromOptions(vision, {
    baseOptions: { modelAssetPath: MODEL.pose, delegate },
    runningMode: 'VIDEO', numPoses: 1,
  });
  setLoading(`判定用のAIを読み込み中（2/3 顔・まばたき・${delegate}）…`);
  face = await mp.FaceLandmarker.createFromOptions(vision, {
    baseOptions: { modelAssetPath: MODEL.face, delegate },
    runningMode: 'VIDEO', numFaces: 1, outputFaceBlendshapes: true,
  });
  setLoading(`判定用のAIを読み込み中（3/3 物の検出・${delegate}）…`);
  obj = await mp.ObjectDetector.createFromOptions(vision, {
    baseOptions: { modelAssetPath: delegate === 'GPU' ? MODEL.objGPU : MODEL.objCPU, delegate },
    runningMode: 'VIDEO', scoreThreshold: 0.25, maxResults: 8,
  });
  S.delegate = delegate;
}

function closeModels() {
  for (const m of [pose, face, obj]) { try { m && m.close(); } catch (_) {} }
  pose = face = obj = null;
}

async function loadModels() {
  const t = performance.now();
  if (!FORCE_CPU) {
    try { await createModels('GPU'); }
    catch (e) { console.warn('GPU failed', e); closeModels(); S.errors.push('GPUで動かなかったのでCPUに切り替え: ' + (e && e.message || e)); }
  }
  if (!pose) await createModels('CPU');
  S.loadMs = performance.now() - t;
  setLoading(null);
}

async function start() {
  $('btnStart').disabled = true;
  try {
    setLoading('カメラを起動中…');
    const stream = await navigator.mediaDevices.getUserMedia({
      video: { facingMode: 'user', width: { ideal: 640 }, height: { ideal: 480 } }, audio: false,
    });
    video.srcObject = stream;
    await video.play();
    await loadModels();
  } catch (e) {
    setLoading(null);
    $('btnStart').disabled = false;
    if (e && e.name === 'NotAllowedError') showError('カメラの使用が許可されていません。iPhone の「設定 → アプリ → Safari → カメラ」を「確認」か「許可」にして、ページを開き直してください。');
    else showError('起動できませんでした: ' + (e && e.message || e));
    return;
  }
  S.started = true;
  S.t0 = performance.now();
  S.lastFrameT = S.t0;
  $('btnStart').hidden = true;
  $('btnTest').disabled = false;
  $('mDelegate').textContent = `${S.delegate}（読み込み ${(S.loadMs / 1000).toFixed(1)}秒）`;
  keepAwake();
  requestAnimationFrame(loop);
}

let wakeLock = null;
async function keepAwake() {
  try { if ('wakeLock' in navigator) wakeLock = await navigator.wakeLock.request('screen'); } catch (_) {}
}
document.addEventListener('visibilitychange', () => {
  if (document.visibilityState === 'visible' && S.started) keepAwake();
});

// ---------- 1フレームの処理 ----------

let lastVideoTime = -1;
function loop() {
  requestAnimationFrame(loop);
  const now = performance.now();
  if (video.readyState < 2 || now - S.lastProc < PROC_MS) return;
  if (video.currentTime === lastVideoTime) return;
  lastVideoTime = video.currentTime;
  S.lastProc = now;
  try {
    processFrame(now);
    S.loopErrors = 0;
  } catch (e) {
    S.loopErrors++;
    console.error(e);
    if (S.loopErrors === 1) showError('判定中のエラー: ' + (e && e.message || e));
    if (S.loopErrors >= 5 && S.delegate === 'GPU') switchToCPU();
  }
}

let switching = false;
async function switchToCPU() {
  if (switching) return;
  switching = true;
  S.started = false;
  closeModels();
  try { await createModels('CPU'); setLoading(null); $('mDelegate').textContent = 'CPU（途中で切り替え）'; }
  catch (e) { showError('CPUでも動きませんでした: ' + (e && e.message || e)); }
  S.loopErrors = 0;
  S.started = true;
  switching = false;
}

function std(arr) {
  const n = arr.length;
  const m = arr.reduce((a, b) => a + b, 0) / n;
  return Math.sqrt(arr.reduce((a, b) => a + (b - m) * (b - m), 0) / n);
}
function median(arr) {
  const s = [...arr].sort((a, b) => a - b);
  return s[Math.floor(s.length / 2)];
}
function trim(win, now, ms) {
  while (win.length && win[0].t < now - ms) win.shift();
}

function processFrame(now) {
  if (!S.started || !pose) return;
  const vw = video.videoWidth, vh = video.videoHeight;
  if (!vw || !vh) return;

  // 体の位置
  const pr = pose.detectForVideo(video, now);
  const lm = pr.landmarks && pr.landmarks[0];
  let present = false, roi = null, hands = [];
  if (lm) {
    const vis = (p) => (p.visibility == null ? 1 : p.visibility);
    const n = lm[0], l = lm[11], r = lm[12];
    present = vis(n) > 0.5 && Math.max(vis(l), vis(r)) > 0.5;
    hands = HAND_POINTS.map((i) => ({ x: lm[i].x * vw, y: lm[i].y * vh, v: vis(lm[i]) }));
    if (present) {
      const sw = Math.hypot((l.x - r.x) * vw, (l.y - r.y) * vh);
      S.poseWin.push({ t: now, p: [n.x * vw, n.y * vh, l.x * vw, l.y * vh, r.x * vw, r.y * vh], sw });
      let x0 = Infinity, y0 = Infinity, x1 = -Infinity, y1 = -Infinity;
      for (let i = 0; i <= 12; i++) {
        x0 = Math.min(x0, lm[i].x * vw); x1 = Math.max(x1, lm[i].x * vw);
        y0 = Math.min(y0, lm[i].y * vh); y1 = Math.max(y1, lm[i].y * vh);
      }
      const w = x1 - x0;
      roi = { x0: x0 - w * 0.25, x1: x1 + w * 0.25, y0: y0 - sw * 0.5, y1: y1 + sw * 0.6 };
    }
  }
  trim(S.poseWin, now, WIN_MS);
  let jit = null;
  if (S.poseWin.length >= 6 && S.poseWin[S.poseWin.length - 1].t - S.poseWin[0].t >= 3000) {
    let sum = 0;
    for (let k = 0; k < 6; k++) sum += std(S.poseWin.map((s) => s.p[k]));
    jit = sum / 6 / median(S.poseWin.map((s) => s.sw));
  }

  // 顔・まばたき
  const fr = face.detectForVideo(video, now);
  const cats = fr.faceBlendshapes && fr.faceBlendshapes[0] && fr.faceBlendshapes[0].categories;
  let blinkAvg = null;
  if (cats) {
    const get = (name) => { const c = cats.find((x) => x.categoryName === name); return c ? c.score : 0; };
    blinkAvg = (get('eyeBlinkLeft') + get('eyeBlinkRight')) / 2;
    if (!S.blinkClosed && blinkAvg > 0.5) { S.blinkClosed = true; S.lastBlink = now; S.blinks++; }
    else if (S.blinkClosed && blinkAvg < 0.3) { S.blinkClosed = false; }
  }
  const faceOn = !!cats;

  // 画素の変化（1秒前との差・体の範囲から背景の変化を引く）
  if (now - S.lastSnapT >= 250) {
    S.lastSnapT = now;
    const W = 80, H = Math.max(1, Math.round(80 * vh / vw));
    if (small.width !== W || small.height !== H) { small.width = W; small.height = H; }
    sctx.drawImage(video, 0, 0, W, H);
    const px = sctx.getImageData(0, 0, W, H).data;
    const gray = new Uint8Array(W * H);
    for (let i = 0, j = 0; i < gray.length; i++, j += 4) gray[i] = (px[j] * 77 + px[j + 1] * 150 + px[j + 2] * 29) >> 8;
    const prev = S.snaps.find((s) => s.t <= now - 900 && s.t >= now - 1500);
    if (prev && roi && prev.gray.length === gray.length) {
      const sx = W / vw, sy = H / vh;
      const rx0 = Math.max(0, Math.floor(roi.x0 * sx)), rx1 = Math.min(W, Math.ceil(roi.x1 * sx));
      const ry0 = Math.max(0, Math.floor(roi.y0 * sy)), ry1 = Math.min(H, Math.ceil(roi.y1 * sy));
      let rs = 0, rn = 0, bs = 0, bn = 0;
      for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
        const i = y * W + x, d = Math.abs(gray[i] - prev.gray[i]);
        if (x >= rx0 && x < rx1 && y >= ry0 && y < ry1) { rs += d; rn++; } else { bs += d; bn++; }
      }
      if (rn > 0) {
        const v = rs / rn - (bn > 20 ? bs / bn : 0);
        S.diffWin.push({ t: now, v: Math.max(0, v) });
      }
    }
    S.snaps.push({ t: now, gray });
    trim(S.snaps, now, 2000);
  }
  trim(S.diffWin, now, WIN_MS);
  const dif = S.diffWin.length >= 8 ? S.diffWin.reduce((a, b) => a + b.v, 0) / S.diffWin.length : null;

  // 物の検出（約1秒に1回）
  if (now - S.lastObjT >= OBJ_MS) {
    S.lastObjT = now;
    const or = obj.detectForVideo(video, now);
    const list = (or.detections || []).map((d) => {
      const c = d.categories[0] || {};
      const b = d.boundingBox || {};
      return { label: c.categoryName || '?', score: c.score || 0, box: { x: b.originX, y: b.originY, w: b.width, h: b.height } };
    });
    S.det = { t: now, list };
    if (list.some((d) => d.label === 'dining table')) S.deskT = now;
    if (S.scene && S.scene.phase === 'run') {
      const st = S.sceneStats[SCENES[S.scene.idx].id];
      for (const d of list) {
        const e = st.labels[d.label] || (st.labels[d.label] = { n: 0, max: 0 });
        e.n++; e.max = Math.max(e.max, d.score);
      }
    }
  }

  // スマホを手に持っているか
  let hold = false, holdBox = null;
  if (now - S.det.t < 1600) {
    for (const d of S.det.list) {
      if (!HOLD_LABELS.includes(d.label) || d.score < P.obj) continue;
      const m = Math.max(d.box.w, d.box.h) * 0.5;
      const near = hands.some((h) => h.x >= d.box.x - m && h.x <= d.box.x + d.box.w + m && h.y >= d.box.y - m && h.y <= d.box.y + d.box.h + m);
      if (near) { hold = true; holdBox = d; break; }
    }
  }

  // 判定
  const blinkRecent = now - S.lastBlink < P.blink * 1000;
  const unknown = jit == null && dif == null;
  const moving = unknown || (jit != null && jit > P.jit) || (dif != null && dif > P.diff) || blinkRecent;
  const raw = !present ? 'away' : hold ? 'phone' : !moving ? 'still' : 'count';
  // 表示の切り替え＝直近 hold 秒のうち7割以上が同じ判定なら（1フレームの途切れでやり直しにしない）
  if (S.disp === null) S.disp = raw;
  S.raw = raw;
  S.rawHist.push({ t: now, raw });
  trim(S.rawHist, now, P.hold * 1000);
  if (S.rawHist.length >= 3 && now - S.rawHist[0].t >= P.hold * 1000 * 0.8) {
    const cnt = {};
    for (const h of S.rawHist) cnt[h.raw] = (cnt[h.raw] || 0) + 1;
    const [best, n] = Object.entries(cnt).sort((a, b) => b[1] - a[1])[0];
    if (best !== S.disp && n / S.rawHist.length >= 0.7) S.disp = best;
  }

  const dt = Math.min(now - S.lastFrameT, 1000);
  S.lastFrameT = now;
  S.totals[S.disp] += dt;
  S.fpsTimes.push(now);
  while (S.fpsTimes.length && S.fpsTimes[0] < now - 2000) S.fpsTimes.shift();
  const fps = S.fpsTimes.length / 2;

  S.cur = { now, present, faceOn, jit, dif, blinkAvg, blinkAgo: (now - S.lastBlink) / 1000, hold, raw, disp: S.disp, fps, roi, hands, lm, holdBox };

  // 1秒ごとの記録
  if (now - S.lastLogT >= 1000) {
    S.lastLogT = now;
    const rec = { t: Math.round((now - S.t0) / 1000), raw, disp: S.disp, jit, dif, blink: blinkRecent, face: faceOn, pose: present, hold, fps };
    if (S.scene && S.scene.phase === 'run') { rec.sc = SCENES[S.scene.idx].id; rec.st = (now - S.scene.runT0) / 1000; }
    S.log.push(rec);
  }
  if (S.scene && S.scene.phase === 'run') {
    const st = S.sceneStats[SCENES[S.scene.idx].id];
    if (S.lastBlink === now) st.blinks++;
  }

  draw(vw, vh);
  updateUI(now);
  tickScene(now);
}

// ---------- 描画 ----------

function draw(vw, vh) {
  if (overlay.width !== vw || overlay.height !== vh) { overlay.width = vw; overlay.height = vh; }
  octx.clearRect(0, 0, vw, vh);
  const mx = (x) => vw - x; // 映像は左右反転で表示しているので、描く位置も反転させる
  const c = S.cur;
  if (c.lm) {
    octx.fillStyle = 'rgba(52,199,89,.9)';
    for (let i = 0; i <= 22; i++) {
      const p = c.lm[i];
      octx.beginPath(); octx.arc(mx(p.x * vw), p.y * vh, i >= 15 ? 4 : 3, 0, Math.PI * 2); octx.fill();
    }
  }
  if (c.roi) {
    octx.strokeStyle = 'rgba(255,255,255,.35)';
    octx.setLineDash([6, 6]);
    octx.strokeRect(mx(c.roi.x1), c.roi.y0, c.roi.x1 - c.roi.x0, c.roi.y1 - c.roi.y0);
    octx.setLineDash([]);
  }
  octx.font = `${Math.max(12, Math.round(vw / 32))}px sans-serif`;
  for (const d of S.det.list) {
    const hot = HOLD_LABELS.includes(d.label);
    octx.strokeStyle = hot ? '#ff453a' : 'rgba(96,165,250,.8)';
    octx.lineWidth = hot ? 3 : 1.5;
    const x = mx(d.box.x + d.box.w);
    octx.strokeRect(x, d.box.y, d.box.w, d.box.h);
    octx.fillStyle = octx.strokeStyle;
    octx.fillText(`${LABEL_JA[d.label] || d.label} ${d.score.toFixed(2)}`, x + 3, d.box.y + 16);
  }
  octx.lineWidth = 1;
}

const fmt = (ms) => {
  const s = Math.floor(ms / 1000);
  const h = Math.floor(s / 3600), m = Math.floor((s % 3600) / 60), r = s % 60;
  return h ? `${h}:${String(m).padStart(2, '0')}:${String(r).padStart(2, '0')}` : `${m}:${String(r).padStart(2, '0')}`;
};

let lastUI = 0;
function updateUI(now) {
  if (now - lastUI < 250) return;
  lastUI = now;
  const c = S.cur;
  const badge = $('badge');
  badge.className = 'badge ' + S.disp;
  badge.textContent = STATE_JA[S.disp];
  $('tCount').textContent = fmt(S.totals.count);
  $('tAway').textContent = fmt(S.totals.away);
  $('tStill').textContent = fmt(S.totals.still);
  $('tPhone').textContent = fmt(S.totals.phone);
  setChip('ckPose', c.present, '肩・顔');
  setChip('ckFace', c.faceOn, '顔（まばたき用）');
  setChip('ckDesk', now - S.deskT < 5000, '机');
  $('mFps').textContent = c.fps.toFixed(1);
  $('mJit').textContent = c.jit == null ? '（集計中）' : `${c.jit.toFixed(4)}（${c.jit > P.jit ? '動きあり' : '動きなし'}）`;
  $('mDiff').textContent = c.dif == null ? '（集計中）' : `${c.dif.toFixed(2)}（${c.dif > P.diff ? '動きあり' : '動きなし'}）`;
  $('mBlink').textContent = `${S.blinks}回・最後から ${isFinite(c.blinkAgo) ? c.blinkAgo.toFixed(0) + '秒' : '—'}${c.blinkAvg == null ? '' : `・目の閉じ ${c.blinkAvg.toFixed(2)}`}`;
  $('mObj').textContent = S.det.list.map((d) => `${LABEL_JA[d.label] || d.label} ${d.score.toFixed(2)}`).join('、') || 'なし';
  $('mRaw').textContent = `${STATE_JA[c.raw]}${c.raw !== S.disp ? `（直近${P.hold}秒の7割になったら切り替え）` : ''}`;
}

function setChip(id, ok, name) {
  const el = $(id);
  el.className = 'chip ' + (ok ? 'ok' : 'ng');
  el.textContent = `${name} ${ok ? '✓' : '×'}`;
}

// ---------- 試験の案内 ----------

function startTest() {
  S.scene = { idx: 0, phase: 'ready' };
  S.results = [];
  S.sceneStats = {};
  $('result').hidden = true;
  $('btnTest').disabled = true;
  showScene();
}

function showScene() {
  const sc = SCENES[S.scene.idx];
  $('sceneCard').hidden = false;
  $('sceneNo').textContent = `${S.scene.idx + 1} / ${SCENES.length}`;
  $('sceneName').textContent = sc.name;
  $('sceneText').textContent = sc.text;
  $('sceneTimer').textContent = `${fmt((sc.sec / SPEED) * 1000)}`;
  $('btnScene').hidden = false;
  $('btnScene').textContent = 'この場面を始める';
  $('btnSkip').hidden = false;
}

function beginScene() {
  const sc = SCENES[S.scene.idx];
  S.scene.phase = 'prep';
  S.scene.prepT0 = performance.now();
  S.sceneStats[sc.id] = { labels: {}, blinks: 0 };
  $('btnScene').hidden = true;
  $('btnSkip').hidden = true;
}

function skipScene() {
  S.results.push({ id: SCENES[S.scene.idx].id, skipped: true });
  nextScene();
}

function nextScene() {
  S.scene.idx++;
  S.scene.phase = 'ready';
  if (S.scene.idx >= SCENES.length) finishTest();
  else showScene();
}

function tickScene(now) {
  if (!S.scene) return;
  const sc = SCENES[S.scene.idx];
  if (!sc) return;
  if (S.scene.phase === 'prep') {
    const left = sc.prep / SPEED * 1000 - (now - S.scene.prepT0);
    $('sceneTimer').textContent = `準備 ${Math.ceil(left / 1000)}秒`;
    if (left <= 0) { S.scene.phase = 'run'; S.scene.runT0 = now; }
  } else if (S.scene.phase === 'run') {
    const left = sc.sec / SPEED * 1000 - (now - S.scene.runT0);
    $('sceneTimer').textContent = `記録中 ${fmt(Math.max(0, left) + 999)}`;
    if (left <= 0) {
      S.results.push(summarize(sc));
      S.scene.phase = 'done';
      $('sceneTimer').textContent = 'この場面は終わりです';
      $('btnScene').hidden = false;
      $('btnScene').textContent = S.scene.idx + 1 < SCENES.length ? '次の場面へ' : '結果を見る';
    }
  }
}

function pct(arr, p) {
  if (!arr.length) return null;
  const s = [...arr].sort((a, b) => a - b);
  return s[Math.min(s.length - 1, Math.floor(p * s.length))];
}

function summarize(sc) {
  const skip = 10 / SPEED;
  const recs = S.log.filter((r) => r.sc === sc.id && r.st >= skip);
  const n = recs.length || 1;
  const share = (k) => Math.round(100 * recs.filter((r) => r.disp === k).length / n);
  const nums = (k) => recs.map((r) => r[k]).filter((v) => v != null);
  const st = S.sceneStats[sc.id] || { labels: {}, blinks: 0 };
  return {
    id: sc.id, name: sc.name, expect: sc.expect, sec: Math.round(sc.sec / SPEED), judged: recs.length,
    ok: Math.round(100 * recs.filter((r) => r.disp === sc.expect).length / n),
    count: share('count'), away: share('away'), still: share('still'), phone: share('phone'),
    rawOk: Math.round(100 * recs.filter((r) => r.raw === sc.expect).length / n),
    jit: [0.1, 0.5, 0.9].map((p) => pct(nums('jit'), p)),
    dif: [0.1, 0.5, 0.9].map((p) => pct(nums('dif'), p)),
    blinks: st.blinks,
    face: Math.round(100 * recs.filter((r) => r.face).length / n),
    pose: Math.round(100 * recs.filter((r) => r.pose).length / n),
    hold: Math.round(100 * recs.filter((r) => r.hold).length / n),
    fps: recs.length ? recs.reduce((a, r) => a + r.fps, 0) / recs.length : 0,
    labels: Object.entries(st.labels).sort((a, b) => b[1].n - a[1].n).slice(0, 6)
      .map(([k, v]) => `${LABEL_JA[k] || k}×${v.n}(${v.max.toFixed(2)})`).join(' '),
    line: recs.map((r) => STATE_LETTER[r.disp]).join(''),
  };
}

function f(v, d) { return v == null ? '-' : v.toFixed(d); }

function resultText() {
  const L = [];
  const d = new Date();
  L.push(`撮影判定テスト ${APP_VER} / ${d.toLocaleString('ja-JP')} / ${S.delegate} / 読込${(S.loadMs / 1000).toFixed(1)}秒 / 映像${video.videoWidth}x${video.videoHeight}${SPEED > 1 ? ` / 速度x${SPEED}` : ''}`);
  L.push(`端末: ${navigator.userAgent}`);
  L.push(`設定: 揺れ>${P.jit} 変化>${P.diff} まばたき${P.blink}秒 切替${P.hold}秒 スマホ>${P.obj}`);
  L.push('（各場面の最初の10秒は除いて集計。正解＝その場面で出るべき判定の割合）');
  for (const r of S.results) {
    if (r.skipped) { L.push(`[${SCENES.find((s) => s.id === r.id).name}] 飛ばした`); continue; }
    L.push(`[${r.name}] ${r.sec}秒(判定${r.judged}秒) 正解${r.ok}%（判定の素${r.rawOk}%） 数え${r.count} 不在${r.away} 静止${r.still} スマホ${r.phone}`);
    L.push(`  揺れ p10/50/90=${f(r.jit[0], 4)}/${f(r.jit[1], 4)}/${f(r.jit[2], 4)} 変化=${f(r.dif[0], 2)}/${f(r.dif[1], 2)}/${f(r.dif[2], 2)} まばたき${r.blinks}回 顔${r.face}% 体${r.pose}% 手にスマホ${r.hold}% fps${r.fps.toFixed(1)}`);
    L.push(`  見えた物: ${r.labels || 'なし'}`);
    L.push(`  時系列: ${r.line}`);
  }
  if (S.errors.length) L.push('エラー: ' + S.errors.join(' / '));
  return L.join('\n');
}

function finishTest() {
  $('sceneCard').hidden = true;
  $('btnTest').disabled = false;
  $('btnTest').textContent = 'もう一度試験する';
  $('result').hidden = false;
  $('resultText').value = resultText();
  S.scene = null;
  $('result').scrollIntoView({ behavior: 'smooth' });
}

async function copyResult() {
  const text = $('resultText').value;
  try {
    await navigator.clipboard.writeText(text);
    $('copyMsg').textContent = 'コピーしました。Claude のチャットに貼ってください。';
  } catch (_) {
    const ta = $('resultText');
    ta.focus(); ta.select(); ta.setSelectionRange(0, text.length);
    $('copyMsg').textContent = '自動でコピーできませんでした。下の文を長押しで全選択してコピーしてください。';
  }
}

// ---------- しきい値 ----------

const TUNE = [['pJit', 'oJit', 'jit', 3], ['pDiff', 'oDiff', 'diff', 1], ['pBlink', 'oBlink', 'blink', 0], ['pHold', 'oHold', 'hold', 0], ['pObj', 'oObj', 'obj', 2]];
function syncTune() {
  for (const [inp, out, key, dg] of TUNE) { $(inp).value = P[key]; $(out).textContent = Number(P[key]).toFixed(dg); }
}
for (const [inp, out, key, dg] of TUNE) {
  $(inp).addEventListener('input', () => {
    P[key] = Number($(inp).value);
    $(out).textContent = P[key].toFixed(dg);
    localStorage.setItem('studycam.params', JSON.stringify(P));
  });
}
$('btnReset').addEventListener('click', () => {
  Object.assign(P, DEFAULTS);
  localStorage.removeItem('studycam.params');
  syncTune();
});
syncTune();

$('btnStart').addEventListener('click', start);
$('btnTest').addEventListener('click', startTest);
$('btnScene').addEventListener('click', () => {
  if (S.scene.phase === 'ready') beginScene();
  else if (S.scene.phase === 'done') nextScene();
});
$('btnSkip').addEventListener('click', () => { if (S.scene && S.scene.phase === 'ready') skipScene(); });
$('btnCopy').addEventListener('click', copyResult);

window.__studycam.start = start;
window.__studycam.startTest = startTest;
window.__studycam.resultText = resultText;
