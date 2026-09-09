/**
 * hyper-engine.js: a small real-time WebGL2 renderer that demonstrates
 * *hypercentric* optics, the browser analog of a HyperEngine-style fly-through.
 *
 * The whole engine is governed by one projection law, taken straight from the
 * article this ships with:
 *
 *     x'(z) = f · x · s(z),   s(z) = 1 / (1 − g·z/Z_c),   g ∈ [−1, 1]
 *
 * where z is *view-space depth* (distance in front of the camera), g is the
 * "vergence" slider, and Z_c is a reference depth beyond the back of the scene.
 *   g = −1  → ordinary perspective (distant things shrink)
 *   g =  0  → orthographic       (size is depth-independent)
 *   g → +1  → hypercentric       (distant things GROW, and their tops/sides
 *                                 swing into view: the "extra faces" effect)
 *
 * Rather than a projection matrix, the vertex shader applies s(z) per vertex
 * (the article's "Approach 1: replace the projection function directly"), with
 * clip.w = 1 so no perspective divide happens. s(z) alone sets on-screen size.
 * Depth is written as a g-independent linear function of true view depth, so
 * occlusion always follows real distance across the whole sweep.
 *
 * Public API:
 *   const api = initHyperEngine(canvas, opts)
 *   opts = {
 *     interactive: true,   // WASD + pointer-lock mouse-look fly camera
 *     autoplay:    false,  // auto-sweep g back and forth
 *     autoCamera:  false,  // slow automatic dolly (for the thumbnail embed)
 *     showMesh:    true,   // draw window.HYPER_MESH if present
 *     onReadout:   fn(state),          // called each time g changes
 *     dom: { slider, playBtn, valOut, regimeOut, cOut } // optional, auto-wired
 *   }
 *   api.setG(g); api.play(); api.pause(); api.destroy();
 *
 * No external dependencies: the little matrix math it needs is inline.
 */
(function (global) {
  'use strict';

  // ---------------------------------------------------------------- mat/vec
  function mat4Identity() {
    return new Float32Array([1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1]);
  }
  function mat4Mul(a, b) { // a·b, column-major
    const o = new Float32Array(16);
    for (let c = 0; c < 4; c++) for (let r = 0; r < 4; r++) {
      o[c*4+r] = a[0*4+r]*b[c*4+0] + a[1*4+r]*b[c*4+1] + a[2*4+r]*b[c*4+2] + a[3*4+r]*b[c*4+3];
    }
    return o;
  }
  function mat4Translate(x, y, z) {
    const m = mat4Identity(); m[12]=x; m[13]=y; m[14]=z; return m;
  }
  function mat4Scale(s) {
    const m = mat4Identity(); m[0]=s; m[5]=s; m[10]=s; return m;
  }
  function mat4RotateY(a) {
    const c = Math.cos(a), s = Math.sin(a), m = mat4Identity();
    m[0]=c; m[2]=-s; m[8]=s; m[10]=c; return m;
  }
  function normalize3(v) {
    const l = Math.hypot(v[0], v[1], v[2]) || 1;
    return [v[0]/l, v[1]/l, v[2]/l];
  }
  function cross3(a, b) {
    return [a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0]];
  }
  function mat4LookAt(eye, center, up) {
    const f = normalize3([center[0]-eye[0], center[1]-eye[1], center[2]-eye[2]]);
    const s = normalize3(cross3(f, up));
    const u = cross3(s, f);
    return new Float32Array([
      s[0], u[0], -f[0], 0,
      s[1], u[1], -f[1], 0,
      s[2], u[2], -f[2], 0,
      -(s[0]*eye[0]+s[1]*eye[1]+s[2]*eye[2]),
      -(u[0]*eye[0]+u[1]*eye[1]+u[2]*eye[2]),
       (f[0]*eye[0]+f[1]*eye[1]+f[2]*eye[2]),
      1
    ]);
  }

  // ---------------------------------------------------------------- shaders
  const VERT = `#version 300 es
  precision highp float;
  in vec3 aPos;
  in vec3 aNormal;
  uniform mat4 uModel, uView;
  uniform float uVergence, uZref, uFocal, uAspect, uNear, uFar;
  out vec3 vViewPos;
  out vec3 vViewNormal;
  out vec3 vWorld;
  void main() {
    vec4 world = uModel * vec4(aPos, 1.0);
    vec4 vp    = uView * world;
    vViewPos    = vp.xyz;
    vWorld      = world.xyz;
    vViewNormal = mat3(uView * uModel) * aNormal;

    float depth = -vp.z;                                  // view-space depth, >0 in front
    // s(z) = 1/(1 - g z / Zc); the single hypercentric projection law.
    float s = (abs(uVergence) < 1e-4) ? 1.0
            : 1.0 / (1.0 - uVergence * depth / uZref);

    float ndcx = uFocal * vp.x * s / uAspect;
    float ndcy = uFocal * vp.y * s;
    // g-independent linear depth so the z-buffer always orders by true distance.
    float ndcz = (depth - uNear) / (uFar - uNear) * 2.0 - 1.0;
    gl_Position = vec4(ndcx, ndcy, ndcz, 1.0);            // w=1 → no perspective divide
  }`;

  const FRAG = `#version 300 es
  precision highp float;
  in vec3 vViewPos;
  in vec3 vViewNormal;
  in vec3 vWorld;
  uniform int  uMode;          // 0 = lit solid, 1 = floor grid
  uniform vec3 uColor;
  uniform vec3 uBg;
  uniform float uNear, uFar, uZref, uGridStep;
  out vec4 outColor;

  float gridLine(vec2 p, float step) {
    vec2 c = p / step;
    vec2 d = abs(fract(c - 0.5) - 0.5) / fwidth(c);
    float line = min(d.x, d.y);
    return 1.0 - clamp(line, 0.0, 1.0);
  }

  void main() {
    float depth = -vViewPos.z;
    // gentle atmospheric fade toward the background, tied to the reference
    // depth and capped so far posts stay visible (they GROW at g→+1).
    float fog = clamp(depth / (uZref * 1.25), 0.0, 0.6);

    vec3 col;
    if (uMode == 1) {
      float g1 = gridLine(vWorld.xz, uGridStep);
      float g2 = gridLine(vWorld.xz, uGridStep * 5.0) * 0.6;
      float g  = max(g1, g2);
      col = mix(uBg, uColor, g);
    } else {
      vec3 N = normalize(vViewNormal);
      vec3 L = normalize(-vViewPos);                      // headlight from the camera
      if (dot(N, L) < 0.0) N = -N;                        // two-sided shading
      float diff = max(dot(N, L), 0.0);
      float rim  = pow(1.0 - max(dot(N, L), 0.0), 3.0) * 0.25;
      col = uColor * (0.28 + 0.72 * diff) + vec3(rim);
    }
    col = mix(col, uBg, fog);
    outColor = vec4(col, 1.0);
  }`;

  function compile(gl, type, src) {
    const sh = gl.createShader(type);
    gl.shaderSource(sh, src.replace(/^\s+/gm, m => m)); // keep as-is
    gl.compileShader(sh);
    if (!gl.getShaderParameter(sh, gl.COMPILE_STATUS)) {
      throw new Error('shader compile: ' + gl.getShaderInfoLog(sh));
    }
    return sh;
  }
  function program(gl, vsrc, fsrc) {
    const p = gl.createProgram();
    gl.attachShader(p, compile(gl, gl.VERTEX_SHADER, vsrc));
    gl.attachShader(p, compile(gl, gl.FRAGMENT_SHADER, fsrc));
    gl.bindAttribLocation(p, 0, 'aPos');
    gl.bindAttribLocation(p, 1, 'aNormal');
    gl.linkProgram(p);
    if (!gl.getProgramParameter(p, gl.LINK_STATUS)) {
      throw new Error('program link: ' + gl.getProgramInfoLog(p));
    }
    return p;
  }

  // ------------------------------------------------------------- geometry
  // Each generator appends interleaved-free parallel arrays: positions, normals
  // (flat xyz) and a Uint32 index list.
  function Builder() { this.pos = []; this.nrm = []; this.idx = []; this.v = 0; }
  Builder.prototype.quad = function (a, b, c, d, n) {
    const base = this.v;
    [a, b, c, d].forEach(p => { this.pos.push(p[0], p[1], p[2]); this.nrm.push(n[0], n[1], n[2]); });
    this.idx.push(base, base+1, base+2, base, base+2, base+3);
    this.v += 4;
  };
  Builder.prototype.box = function (cx, cy, cz, hx, hy, hz) {
    const x0=cx-hx, x1=cx+hx, y0=cy-hy, y1=cy+hy, z0=cz-hz, z1=cz+hz;
    this.quad([x0,y0,z1],[x1,y0,z1],[x1,y1,z1],[x0,y1,z1], [0,0,1]);   // +z
    this.quad([x1,y0,z0],[x0,y0,z0],[x0,y1,z0],[x1,y1,z0], [0,0,-1]);  // -z
    this.quad([x1,y0,z1],[x1,y0,z0],[x1,y1,z0],[x1,y1,z1], [1,0,0]);   // +x
    this.quad([x0,y0,z0],[x0,y0,z1],[x0,y1,z1],[x0,y1,z0], [-1,0,0]);  // -x
    this.quad([x0,y1,z1],[x1,y1,z1],[x1,y1,z0],[x0,y1,z0], [0,1,0]);   // +y (top)
    this.quad([x0,y0,z0],[x1,y0,z0],[x1,y0,z1],[x0,y0,z1], [0,-1,0]);  // -y
  };
  Builder.prototype.finish = function () {
    return {
      pos: new Float32Array(this.pos),
      nrm: new Float32Array(this.nrm),
      idx: new Uint32Array(this.idx)
    };
  };

  function b64ToBuffer(b64) {
    const bin = atob(b64), len = bin.length, bytes = new Uint8Array(len);
    for (let i = 0; i < len; i++) bytes[i] = bin.charCodeAt(i);
    return bytes.buffer;
  }

  // -------------------------------------------------------------- CSS colors
  function cssColor(name, fallback) {
    const v = getComputedStyle(document.documentElement).getPropertyValue(name).trim();
    return v || fallback;
  }
  function hexToRgb(hex) {
    hex = hex.replace('#', '');
    if (hex.length === 3) hex = hex.split('').map(c => c + c).join('');
    const n = parseInt(hex, 16);
    return [((n>>16)&255)/255, ((n>>8)&255)/255, (n&255)/255];
  }

  // ------------------------------------------------------------------ engine
  function initHyperEngine(canvas, opts) {
    opts = opts || {};
    const gl = canvas.getContext('webgl2', { antialias: true, alpha: false });
    if (!gl) { console.warn('WebGL2 unavailable'); return null; }

    const prog = program(gl, VERT, FRAG);
    const U = {};
    ['uModel','uView','uVergence','uZref','uFocal','uAspect','uNear','uFar',
     'uMode','uColor','uBg','uGridStep'].forEach(n => U[n] = gl.getUniformLocation(prog, n));

    // ---- colors from the site theme ----
    const bg     = hexToRgb(cssColor('--canvas-bg', '#f3f1ec'));
    const accent = hexToRgb(cssColor('--accent', '#2f5fae'));
    const rule   = hexToRgb(cssColor('--fg-faint', '#8a8377'));
    const postCol = accent;
    const meshCol = hexToRgb(cssColor('--accent', '#c4553b'));

    // ---- scene scale keyed to the mesh radius ----
    const mesh = (opts.showMesh !== false && global.HYPER_MESH) ? global.HYPER_MESH : null;
    const R = (mesh && mesh.radius) ? mesh.radius : 10;
    const floorY = -R;
    const NEAR = 0.05 * R, FAR = 42 * R, ZREF = 22 * R;
    // With clip.w = 1 the shader bakes s(z) directly, so `focal` maps raw view
    // units to NDC. It must scale like 1/R for the scene to frame regardless
    // of mesh size (it is NOT a tan(fov/2) term).
    let focal = 0.62 / R;

    // ---- build post avenue + floor ----
    // A central avenue of equal-size posts. They sit near the sight line so
    // that the whole depth range stays in frame across the g sweep (far posts
    // swing outward and grow as g→+1, the hypercentric size flip in 3D).
    const postB = new Builder();
    const rows = [-0.55 * R, 0.55 * R];
    for (const x of rows) {
      for (let i = 0; i < 8; i++) {
        const z = -0.6 * R - i * 1.1 * R;
        postB.box(x, floorY + 0.8 * R, z, 0.11 * R, 0.8 * R, 0.11 * R);
      }
    }
    const posts = postB.finish();

    const floorB = new Builder();
    floorB.quad([-30*R, floorY, 6*R], [30*R, floorY, 6*R],
                [30*R, floorY, -42*R], [-30*R, floorY, -42*R], [0,1,0]);
    const floor = floorB.finish();

    // ---- GPU buffers ----
    function makeVAO(geo) {
      const vao = gl.createVertexArray();
      gl.bindVertexArray(vao);
      const pb = gl.createBuffer();
      gl.bindBuffer(gl.ARRAY_BUFFER, pb);
      gl.bufferData(gl.ARRAY_BUFFER, geo.pos, gl.STATIC_DRAW);
      gl.enableVertexAttribArray(0);
      gl.vertexAttribPointer(0, 3, gl.FLOAT, false, 0, 0);
      const nb = gl.createBuffer();
      gl.bindBuffer(gl.ARRAY_BUFFER, nb);
      gl.bufferData(gl.ARRAY_BUFFER, geo.nrm, gl.STATIC_DRAW);
      gl.enableVertexAttribArray(1);
      gl.vertexAttribPointer(1, 3, gl.FLOAT, false, 0, 0);
      const ib = gl.createBuffer();
      gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, ib);
      gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, geo.idx, gl.STATIC_DRAW);
      gl.bindVertexArray(null);
      return { vao, count: geo.idx.length };
    }

    const postVAO = makeVAO(posts);
    const floorVAO = makeVAO(floor);

    let meshVAO = null;
    if (mesh) {
      meshVAO = makeVAO({
        pos: new Float32Array(b64ToBuffer(mesh.positions)),
        nrm: new Float32Array(b64ToBuffer(mesh.normals)),
        idx: new Uint32Array(b64ToBuffer(mesh.indices))
      });
    }

    // ---- camera state (fly) ----
    const cam = {
      pos: [0, 0.6 * R, 3.0 * R],
      yaw: -Math.PI / 2,      // looking down -z
      pitch: -0.16            // slight downward tilt for a 3/4 view of the avenue
    };
    function forward() {
      const cp = Math.cos(cam.pitch);
      return [cp * Math.cos(cam.yaw), Math.sin(cam.pitch), cp * Math.sin(cam.yaw)];
    }

    // ---- interaction ----
    const keys = Object.create(null);
    const interactive = opts.interactive !== false;
    let disposed = false;
    const listeners = [];
    function on(el, ev, fn, o) { el.addEventListener(ev, fn, o); listeners.push([el, ev, fn, o]); }

    if (interactive) {
      on(window, 'keydown', e => {
        const k = e.key.toLowerCase();
        if (['w','a','s','d','q','e',' '].includes(k)) {
          keys[k] = true;
          if (document.pointerLockElement === canvas) e.preventDefault();
        }
      });
      on(window, 'keyup', e => { keys[e.key.toLowerCase()] = false; });
      on(canvas, 'click', () => { if (canvas.requestPointerLock) canvas.requestPointerLock(); });
      on(document, 'mousemove', e => {
        if (document.pointerLockElement !== canvas) return;
        cam.yaw   += e.movementX * 0.0025;
        cam.pitch -= e.movementY * 0.0025;
        cam.pitch = Math.max(-1.4, Math.min(1.4, cam.pitch));
      });
      // touch look-around
      let tx = 0, ty = 0, touching = false;
      on(canvas, 'touchstart', e => { touching = true; tx = e.touches[0].clientX; ty = e.touches[0].clientY; }, {passive:true});
      on(canvas, 'touchmove', e => {
        if (!touching) return;
        const t = e.touches[0];
        cam.yaw   += (t.clientX - tx) * 0.006;
        cam.pitch = Math.max(-1.4, Math.min(1.4, cam.pitch - (t.clientY - ty) * 0.006));
        tx = t.clientX; ty = t.clientY;
      }, {passive:true});
      on(canvas, 'touchend', () => { touching = false; });
    }

    // ---- vergence g state ----
    let g = (typeof opts.g === 'number') ? opts.g : -1;
    let playing = !!opts.autoplay;
    let sweepT = 0;
    const dom = opts.dom || {};

    function regimeName(gv) {
      if (gv < -0.03) return 'Perspective';
      if (gv >  0.03) return 'Hypercentric';
      return 'Orthographic (telecentric)';
    }
    function convergenceDepth(gv) {
      if (Math.abs(gv) < 1e-4) return Infinity;
      return ZREF / gv;
    }
    function syncReadout() {
      if (dom.valOut)    dom.valOut.textContent = g.toFixed(2);
      if (dom.regimeOut) dom.regimeOut.textContent = regimeName(g);
      if (dom.cOut) {
        const c = convergenceDepth(g);
        dom.cOut.textContent = !Number.isFinite(c) ? 'infinity (parallel rays)'
          : c < 0 ? (c / R).toFixed(1) + ' units behind the camera'
                  : (c / R).toFixed(1) + ' units, beyond the scene';
      }
      if (dom.slider) dom.slider.value = String(Math.round(g * 100));
      if (typeof opts.onReadout === 'function') {
        opts.onReadout({ g, regime: regimeName(g), c: convergenceDepth(g) });
      }
    }

    if (dom.slider) {
      on(dom.slider, 'input', () => {
        g = parseInt(dom.slider.value, 10) / 100;
        playing = false;
        if (dom.playBtn) dom.playBtn.textContent = 'Play';
        syncReadout();
      });
    }
    if (dom.playBtn) {
      on(dom.playBtn, 'click', () => {
        playing = !playing;
        dom.playBtn.textContent = playing ? 'Pause' : 'Play';
      });
    }

    // ---- resize (retina aware) ----
    function resize() {
      const dpr = Math.min(global.devicePixelRatio || 1, 2);
      const w = canvas.clientWidth || canvas.width;
      const h = canvas.clientHeight || canvas.height;
      const W = Math.round(w * dpr), H = Math.round(h * dpr);
      if (canvas.width !== W || canvas.height !== H) { canvas.width = W; canvas.height = H; }
    }
    on(window, 'resize', resize);

    // ---- render loop ----
    gl.enable(gl.DEPTH_TEST);
    const reduceMotion = !!(global.matchMedia && global.matchMedia('(prefers-reduced-motion: reduce)').matches);
    let last = null;

    function drawGeo(vao, model, mode, color, gridStep) {
      gl.bindVertexArray(vao.vao);
      gl.uniformMatrix4fv(U.uModel, false, model);
      gl.uniform1i(U.uMode, mode);
      gl.uniform3fv(U.uColor, color);
      gl.uniform1f(U.uGridStep, gridStep || R);
      gl.drawElements(gl.TRIANGLES, vao.count, gl.UNSIGNED_INT, 0);
    }

    function frame(ts) {
      if (disposed) return;
      if (last === null) last = ts;
      const dt = Math.min((ts - last) / 1000, 0.05);
      last = ts;

      // camera movement (WASD)
      if (interactive) {
        const f = forward();
        const right = normalize3(cross3(f, [0,1,0]));
        const spd = 4.5 * R * dt;
        if (keys['w']) { cam.pos[0]+=f[0]*spd; cam.pos[1]+=f[1]*spd; cam.pos[2]+=f[2]*spd; }
        if (keys['s']) { cam.pos[0]-=f[0]*spd; cam.pos[1]-=f[1]*spd; cam.pos[2]-=f[2]*spd; }
        if (keys['d']) { cam.pos[0]+=right[0]*spd; cam.pos[2]+=right[2]*spd; }
        if (keys['a']) { cam.pos[0]-=right[0]*spd; cam.pos[2]-=right[2]*spd; }
        if (keys['e'] || keys[' ']) cam.pos[1] += spd;
        if (keys['q']) cam.pos[1] -= spd;
      }
      // gentle automatic dolly for the thumbnail embed
      if (opts.autoCamera && !reduceMotion) {
        sweepT += dt;
        cam.pos[2] = 3.4 * R - 1.1 * R * (0.5 - 0.5 * Math.cos(sweepT * 0.35));
        cam.yaw = -Math.PI / 2 + Math.sin(sweepT * 0.18) * 0.18;
      }

      // vergence auto-sweep
      if (playing && !reduceMotion) {
        sweepT += dt;
        const period = 7;
        const phase = (sweepT % period) / period;
        g = Math.sin(phase * Math.PI * 2 - Math.PI / 2);
        syncReadout();
      }

      resize();
      gl.viewport(0, 0, canvas.width, canvas.height);
      gl.clearColor(bg[0], bg[1], bg[2], 1);
      gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);

      const f = forward();
      const view = mat4LookAt(cam.pos, [cam.pos[0]+f[0], cam.pos[1]+f[1], cam.pos[2]+f[2]], [0,1,0]);

      gl.useProgram(prog);
      gl.uniformMatrix4fv(U.uView, false, view);
      gl.uniform1f(U.uVergence, g);
      gl.uniform1f(U.uZref, ZREF);
      gl.uniform1f(U.uFocal, focal);
      gl.uniform1f(U.uAspect, canvas.width / canvas.height);
      gl.uniform1f(U.uNear, NEAR);
      gl.uniform1f(U.uFar, FAR);
      gl.uniform3fv(U.uBg, bg);

      drawGeo(floorVAO, mat4Identity(), 1, rule, R);
      drawGeo(postVAO, mat4Identity(), 0, postCol, R);
      if (meshVAO) {
        // spin slowly so the "extra faces" reveal reads as geometry, not a trick
        const spin = reduceMotion ? 0.6 : ts * 0.00025;
        const model = mat4Mul(mat4Translate(0, 0, -1.1 * R),
                       mat4Mul(mat4RotateY(spin), mat4Scale(1)));
        drawGeo(meshVAO, model, 0, meshCol, R);
      }

      requestAnimationFrame(frame);
    }

    syncReadout();
    resize();
    requestAnimationFrame(frame);

    return {
      setG(v) { g = Math.max(-1, Math.min(1, v)); playing = false; syncReadout(); },
      play()  { playing = true; if (dom.playBtn) dom.playBtn.textContent = 'Pause'; },
      pause() { playing = false; if (dom.playBtn) dom.playBtn.textContent = 'Play'; },
      setFocal(v) { focal = v; },
      get state() { return { g, playing, regime: regimeName(g) }; },
      destroy() {
        disposed = true;
        listeners.forEach(([el, ev, fn, o]) => el.removeEventListener(ev, fn, o));
      }
    };
  }

  global.initHyperEngine = initHyperEngine;
})(window);
