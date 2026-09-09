/* ---------------------------------------------------------------------
   fluid.js: the live particle fluid, taken straight from my
   riyasinghrathore.github.io front page (same shaders, same solver,
   same tweakpane controls, same four render modes). Two changes for
   this article: the draggable bio card is gone (the writing sits on top
   instead), and the pointer is listened for on the whole window so the
   fluid can be stirred from anywhere over the page.

   Built on the gpu-io library; its MIT license notice lives at the top
   of assets/libs/gpu-io.js.
   --------------------------------------------------------------------- */
(function () {
    'use strict';

    if (typeof GPUIO === 'undefined' || !document.getElementById('simulation-container')) return;

    const {
        GPUComposer,
        GPUProgram,
        GPULayer,
        SHORT,
        INT,
        FLOAT,
        REPEAT,
        NEAREST,
        LINEAR,
        GLSL1,
        GLSL3,
        WEBGL1,
        WEBGL2,
        isWebGL2Supported,
        renderSignedAmplitudeProgram,
    } = GPUIO;

    const PARAMS = {
        // Render settings
        trailLength: 15,
        render: 'Fluid',

        // Fluid dynamics parameters
        viscosity: 0.98,           // Velocity decay (0.9 = high viscosity, 1.0 = inviscid)
        forceScale: 2.0,           // Touch force multiplier
        forceRadius: 30,           // Touch influence radius
        jacobiIterations: 3,       // Pressure solver iterations
        velocityScale: 8,          // Velocity field resolution scale
        maxVelocity: 30,           // Velocity clamp

        // Particle settings
        particleDensity: 0.1,      // Particles per pixel
        particleLifetime: 1000,    // Frames before reset
        numRenderSteps: 3,         // Sub-steps per frame
    };

    // Simulation constants
    const TOUCH_FORCE_SCALE = 2;
    const PARTICLE_DENSITY = 0.1;
    const MAX_NUM_PARTICLES = 100000;
    const PARTICLE_LIFETIME = 1000;
    const PRESSURE_CALC_ALPHA = -1;
    const PRESSURE_CALC_BETA = 0.25;
    const NUM_RENDER_STEPS = 3;
    const VELOCITY_SCALE_FACTOR = 8;
    const MAX_VELOCITY = 30;
    const POSITION_NUM_COMPONENTS = 4;

    const container = document.getElementById('simulation-container');
    const controlsContainer = document.getElementById('controls-container');

    // Create canvas
    const canvas = document.createElement('canvas');
    container.appendChild(canvas);

    // Initialize simulation controls pane
    const controlsPane = new Tweakpane.Pane({
        container: controlsContainer,
        title: 'Simulation Controls',
        expanded: true,
    });

    // Determine WebGL/GLSL version
    const contextID = isWebGL2Supported() ? WEBGL2 : WEBGL1;
    const glslVersion = isWebGL2Supported() ? GLSL3 : GLSL1;

    function calcNumParticles(width, height) {
        return Math.min(Math.ceil(width * height * PARTICLE_DENSITY), MAX_NUM_PARTICLES);
    }

    function getContainerSize() {
        return {
            width: window.innerWidth,
            height: window.innerHeight
        };
    }

    const { width: initialWidth, height: initialHeight } = getContainerSize();
    canvas.width = initialWidth;
    canvas.height = initialHeight;

    let NUM_PARTICLES = calcNumParticles(canvas.width, canvas.height);

    const composer = new GPUComposer({ canvas, contextID, glslVersion });

    // Initialize state layers
    const velocityState = new GPULayer(composer, {
        name: 'velocity',
        dimensions: [Math.ceil(initialWidth / VELOCITY_SCALE_FACTOR), Math.ceil(initialHeight / VELOCITY_SCALE_FACTOR)],
        type: FLOAT,
        filter: LINEAR,
        numComponents: 2,
        wrapX: REPEAT,
        wrapY: REPEAT,
        numBuffers: 2,
    });

    const divergenceState = new GPULayer(composer, {
        name: 'divergence',
        dimensions: [velocityState.width, velocityState.height],
        type: FLOAT,
        filter: NEAREST,
        numComponents: 1,
        wrapX: REPEAT,
        wrapY: REPEAT,
    });

    const pressureState = new GPULayer(composer, {
        name: 'pressure',
        dimensions: [velocityState.width, velocityState.height],
        type: FLOAT,
        filter: NEAREST,
        numComponents: 1,
        wrapX: REPEAT,
        wrapY: REPEAT,
        numBuffers: 2,
    });

    const particlePositionState = new GPULayer(composer, {
        name: 'position',
        dimensions: NUM_PARTICLES,
        type: FLOAT,
        numComponents: POSITION_NUM_COMPONENTS,
        numBuffers: 2,
    });

    const particleInitialState = new GPULayer(composer, {
        name: 'initialPosition',
        dimensions: NUM_PARTICLES,
        type: FLOAT,
        numComponents: POSITION_NUM_COMPONENTS,
        numBuffers: 1,
    });

    const particleAgeState = new GPULayer(composer, {
        name: 'age',
        dimensions: NUM_PARTICLES,
        type: SHORT,
        numComponents: 1,
        numBuffers: 2,
    });

    const trailState = new GPULayer(composer, {
        name: 'trails',
        dimensions: [canvas.width, canvas.height],
        type: FLOAT,
        filter: NEAREST,
        numComponents: 1,
        numBuffers: 2,
    });

    // Suminagashi color state - stores RGB colors that advect with fluid
    const colorState = new GPULayer(composer, {
        name: 'colors',
        dimensions: [canvas.width, canvas.height],
        type: FLOAT,
        filter: LINEAR,
        numComponents: 4, // RGBA
        numBuffers: 2,
        clearValue: [0.98, 0.96, 0.94, 1], // Cream background
    });

    // Initialize programs
    const advection = new GPUProgram(composer, {
        name: 'advection',
        fragmentShader: `
        in vec2 v_uv;

        uniform sampler2D u_state;
        uniform sampler2D u_velocity;
        uniform vec2 u_dimensions;

        out vec2 out_state;

        void main() {
            out_state = texture(u_state, v_uv - texture(u_velocity, v_uv).xy / u_dimensions).xy;
        }`,
        uniforms: [
            { name: 'u_state', value: 0, type: INT },
            { name: 'u_velocity', value: 1, type: INT },
            { name: 'u_dimensions', value: [canvas.width, canvas.height], type: FLOAT },
        ],
    });

    const divergence2D = new GPUProgram(composer, {
        name: 'divergence2D',
        fragmentShader: `
        in vec2 v_uv;

        uniform sampler2D u_vectorField;
        uniform vec2 u_pxSize;

        out float out_divergence;

        void main() {
            float n = texture(u_vectorField, v_uv + vec2(0, u_pxSize.y)).y;
            float s = texture(u_vectorField, v_uv - vec2(0, u_pxSize.y)).y;
            float e = texture(u_vectorField, v_uv + vec2(u_pxSize.x, 0)).x;
            float w = texture(u_vectorField, v_uv - vec2(u_pxSize.x, 0)).x;
            out_divergence = 0.5 * (e - w + n - s);
        }`,
        uniforms: [
            { name: 'u_vectorField', value: 0, type: INT },
            { name: 'u_pxSize', value: [1 / velocityState.width, 1 / velocityState.height], type: FLOAT },
        ],
    });

    const jacobi = new GPUProgram(composer, {
        name: 'jacobi',
        fragmentShader: `
        in vec2 v_uv;

        uniform float u_alpha;
        uniform float u_beta;
        uniform vec2 u_pxSize;
        uniform sampler2D u_previousState;
        uniform sampler2D u_divergence;

        out vec4 out_jacobi;

        void main() {
            vec4 n = texture(u_previousState, v_uv + vec2(0, u_pxSize.y));
            vec4 s = texture(u_previousState, v_uv - vec2(0, u_pxSize.y));
            vec4 e = texture(u_previousState, v_uv + vec2(u_pxSize.x, 0));
            vec4 w = texture(u_previousState, v_uv - vec2(u_pxSize.x, 0));
            vec4 d = texture(u_divergence, v_uv);
            out_jacobi = (n + s + e + w + u_alpha * d) * u_beta;
        }`,
        uniforms: [
            { name: 'u_alpha', value: PRESSURE_CALC_ALPHA, type: FLOAT },
            { name: 'u_beta', value: PRESSURE_CALC_BETA, type: FLOAT },
            { name: 'u_pxSize', value: [1 / velocityState.width, 1 / velocityState.height], type: FLOAT },
            { name: 'u_previousState', value: 0, type: INT },
            { name: 'u_divergence', value: 1, type: INT },
        ],
    });

    const gradientSubtraction = new GPUProgram(composer, {
        name: 'gradientSubtraction',
        fragmentShader: `
        in vec2 v_uv;

        uniform vec2 u_pxSize;
        uniform sampler2D u_scalarField;
        uniform sampler2D u_vectorField;

        out vec2 out_result;

        void main() {
            float n = texture(u_scalarField, v_uv + vec2(0, u_pxSize.y)).r;
            float s = texture(u_scalarField, v_uv - vec2(0, u_pxSize.y)).r;
            float e = texture(u_scalarField, v_uv + vec2(u_pxSize.x, 0)).r;
            float w = texture(u_scalarField, v_uv - vec2(u_pxSize.x, 0)).r;
            out_result = texture(u_vectorField, v_uv).xy - 0.5 * vec2(e - w, n - s);
        }`,
        uniforms: [
            { name: 'u_pxSize', value: [1 / velocityState.width, 1 / velocityState.height], type: FLOAT },
            { name: 'u_scalarField', value: 0, type: INT },
            { name: 'u_vectorField', value: 1, type: INT },
        ],
    });

    // Viscosity/decay - applies velocity damping
    const applyViscosity = new GPUProgram(composer, {
        name: 'applyViscosity',
        fragmentShader: `
        in vec2 v_uv;

        uniform sampler2D u_velocity;
        uniform float u_decay;
        uniform float u_maxVelocity;

        out vec2 out_velocity;

        void main() {
            vec2 vel = texture(u_velocity, v_uv).xy * u_decay;
            float mag = length(vel);
            if (mag > u_maxVelocity) {
                vel = vel / mag * u_maxVelocity;
            }
            out_velocity = vel;
        }`,
        uniforms: [
            { name: 'u_velocity', value: 0, type: INT },
            { name: 'u_decay', value: PARAMS.viscosity, type: FLOAT },
            { name: 'u_maxVelocity', value: PARAMS.maxVelocity, type: FLOAT },
        ],
    });

    const renderParticles = new GPUProgram(composer, {
        name: 'renderParticles',
        fragmentShader: `
        #define FADE_TIME 0.1

        in vec2 v_uv;
        in vec2 v_uv_position;

        uniform isampler2D u_ages;
        uniform sampler2D u_velocity;

        out float out_state;

        void main() {
            float ageFraction = float(texture(u_ages, v_uv_position).x) / ${PARTICLE_LIFETIME.toFixed(1)};
            float opacity = mix(0.0, 1.0, min(ageFraction * 10.0, 1.0)) * mix(1.0, 0.0, max(ageFraction * 10.0 - 90.0, 0.0));
            vec2 velocity = texture(u_velocity, v_uv).xy;
            float multiplier = clamp(dot(velocity, velocity) * 0.05 + 0.7, 0.0, 1.0);
            out_state = opacity * multiplier;
        }`,
        uniforms: [
            { name: 'u_ages', value: 0, type: INT },
            { name: 'u_velocity', value: 1, type: INT },
        ],
    });

    const ageParticles = new GPUProgram(composer, {
        name: 'ageParticles',
        fragmentShader: `
        in vec2 v_uv;

        uniform isampler2D u_ages;

        out int out_age;

        void main() {
            int age = texture(u_ages, v_uv).x + 1;
            out_age = stepi(age, ${PARTICLE_LIFETIME}) * age;
        }`,
        uniforms: [
            { name: 'u_ages', value: 0, type: INT },
        ],
    });

    const advectParticles = new GPUProgram(composer, {
        name: 'advectParticles',
        fragmentShader: `
        in vec2 v_uv;

        uniform vec2 u_dimensions;
        uniform sampler2D u_positions;
        uniform sampler2D u_velocity;
        uniform isampler2D u_ages;
        uniform sampler2D u_initialPositions;

        out vec4 out_position;

        void main() {
            vec4 positionData = texture(u_positions, v_uv);
            vec2 absolute = positionData.rg;
            vec2 displacement = positionData.ba;
            vec2 position = absolute + displacement;

            vec2 pxSize = 1.0 / u_dimensions;
            vec2 velocity1 = texture(u_velocity, position * pxSize).xy;
            vec2 halfStep = position + velocity1 * 0.5 * ${1 / NUM_RENDER_STEPS};
            vec2 velocity2 = texture(u_velocity, halfStep * pxSize).xy;
            displacement += velocity2 * ${1 / NUM_RENDER_STEPS};

            float shouldMerge = step(20.0, dot(displacement, displacement));
            absolute = mod(absolute + shouldMerge * displacement + u_dimensions, u_dimensions);
            displacement *= (1.0 - shouldMerge);

            int shouldReset = stepi(texture(u_ages, v_uv).x, 1);
            out_position = mix(vec4(absolute, displacement), texture(u_initialPositions, v_uv), float(shouldReset));
        }`,
        uniforms: [
            { name: 'u_positions', value: 0, type: INT },
            { name: 'u_velocity', value: 1, type: INT },
            { name: 'u_ages', value: 2, type: INT },
            { name: 'u_initialPositions', value: 3, type: INT },
            { name: 'u_dimensions', value: [canvas.width, canvas.height], type: FLOAT },
        ],
    });

    const fadeTrails = new GPUProgram(composer, {
        name: 'fadeTrails',
        fragmentShader: `
        in vec2 v_uv;

        uniform sampler2D u_image;
        uniform float u_increment;

        out float out_color;

        void main() {
            out_color = max(texture(u_image, v_uv).x + u_increment, 0.0);
        }`,
        uniforms: [
            { name: 'u_image', value: 0, type: INT },
            { name: 'u_increment', value: -1 / PARAMS.trailLength, type: FLOAT },
        ],
    });

    const renderTrails = new GPUProgram(composer, {
        name: 'renderTrails',
        fragmentShader: `
            in vec2 v_uv;
            uniform sampler2D u_trailState;
            out vec4 out_color;
            void main() {
                vec3 background = vec3(0.98, 0.965, 0.94);
                vec3 particle = vec3(0.04, 0.06, 0.11);
                out_color = vec4(mix(background, particle, texture(u_trailState, v_uv).x), 1);
            }
        `,
    });

    const renderPressure = renderSignedAmplitudeProgram(composer, {
        name: 'renderPressure',
        type: pressureState.type,
        scale: 0.5,
        component: 'x',
    });

    // Suminagashi: advect colors with velocity field
    const advectColors = new GPUProgram(composer, {
        name: 'advectColors',
        fragmentShader: `
        in vec2 v_uv;

        uniform sampler2D u_colors;
        uniform sampler2D u_velocity;
        uniform vec2 u_dimensions;

        out vec4 out_color;

        void main() {
            vec2 velocity = texture(u_velocity, v_uv).xy;
            vec2 advectedUV = v_uv - velocity / u_dimensions;
            out_color = texture(u_colors, advectedUV);
        }`,
        uniforms: [
            { name: 'u_colors', value: 0, type: INT },
            { name: 'u_velocity', value: 1, type: INT },
            { name: 'u_dimensions', value: [canvas.width, canvas.height], type: FLOAT },
        ],
    });

    // Suminagashi: drop ink at a position - alternating black and blue
    const dropInk = new GPUProgram(composer, {
        name: 'dropInk',
        fragmentShader: `
        in vec2 v_uv;
        in vec2 v_uv_local;

        uniform sampler2D u_colors;
        uniform float u_time;

        out vec4 out_color;

        // Suminagashi ink colors - black and blue that blend into each other
        vec3 getInkColor(vec2 uv, float t) {
            // Rich black ink
            vec3 black = vec3(0.02, 0.02, 0.03);
            // Deep indigo/blue ink
            vec3 blue = vec3(0.05, 0.15, 0.35);
            // Lighter blue for highlights
            vec3 lightBlue = vec3(0.12, 0.28, 0.52);

            // Create swirling pattern between black and blue
            float wave1 = sin(uv.x * 8.0 + uv.y * 4.0 + t * 2.0) * 0.5 + 0.5;
            float wave2 = cos(uv.y * 6.0 - uv.x * 3.0 + t * 1.5) * 0.5 + 0.5;
            float wave3 = sin((uv.x + uv.y) * 5.0 + t * 0.8) * 0.5 + 0.5;

            // Blend waves for organic feel
            float blend = wave1 * 0.5 + wave2 * 0.3 + wave3 * 0.2;

            // Mix between black and blue based on blend
            vec3 color = mix(black, blue, blend);
            // Add occasional lighter blue highlights
            color = mix(color, lightBlue, wave3 * wave1 * 0.4);

            return color;
        }

        void main() {
            vec4 currentColor = texture(u_colors, v_uv);

            // Circular ink drop with soft edges
            float dist = length(v_uv_local * 2.0 - 1.0);
            float ink = smoothstep(1.0, 0.15, dist);

            vec3 inkColor = getInkColor(v_uv, u_time);
            out_color = mix(currentColor, vec4(inkColor, 1.0), ink * 0.75);
        }`,
        uniforms: [
            { name: 'u_colors', value: 0, type: INT },
            { name: 'u_time', value: 0, type: FLOAT },
        ],
    });

    // Suminagashi: render the colors to screen
    const renderMarbling = new GPUProgram(composer, {
        name: 'renderMarbling',
        fragmentShader: `
        in vec2 v_uv;
        uniform sampler2D u_colors;
        out vec4 out_color;
        void main() {
            out_color = texture(u_colors, v_uv);
        }`,
        uniforms: [
            { name: 'u_colors', value: 0, type: INT },
        ],
    });

    // Initialize suminagashi with a subtle texture pattern
    const initMarbleTexture = new GPUProgram(composer, {
        name: 'initMarbleTexture',
        fragmentShader: `
        in vec2 v_uv;

        out vec4 out_color;

        // Simple noise function
        float hash(vec2 p) {
            return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453);
        }

        float noise(vec2 p) {
            vec2 i = floor(p);
            vec2 f = fract(p);
            f = f * f * (3.0 - 2.0 * f);

            float a = hash(i);
            float b = hash(i + vec2(1.0, 0.0));
            float c = hash(i + vec2(0.0, 1.0));
            float d = hash(i + vec2(1.0, 1.0));

            return mix(mix(a, b, f.x), mix(c, d, f.x), f.y);
        }

        void main() {
            // Cream/paper background with subtle texture
            vec3 baseColor = vec3(0.96, 0.94, 0.90);

            // Add noise-based texture variation
            float n1 = noise(v_uv * 8.0);
            float n2 = noise(v_uv * 16.0 + 100.0);
            float n3 = noise(v_uv * 32.0 + 200.0);

            // Combine noise at different frequencies
            float pattern = n1 * 0.5 + n2 * 0.3 + n3 * 0.2;

            // Subtle swirl hints
            float swirl = sin(v_uv.x * 4.0 + v_uv.y * 3.0) * 0.5 + 0.5;

            // Mix in some very subtle color variation
            vec3 tint1 = vec3(0.94, 0.92, 0.88); // Warm
            vec3 tint2 = vec3(0.92, 0.93, 0.95); // Cool

            vec3 color = mix(tint1, tint2, swirl * pattern);
            color = mix(color, baseColor, 0.7);

            // Add subtle dark veins
            float vein = smoothstep(0.48, 0.52, noise(v_uv * 3.0 + vec2(noise(v_uv * 2.0))));
            color = mix(color, vec3(0.88, 0.86, 0.82), vein * 0.15);

            out_color = vec4(color, 1.0);
        }`,
        uniforms: [],
    });

    // Touch interaction program
    const touch = new GPUProgram(composer, {
        name: 'touch',
        fragmentShader: `
        in vec2 v_uv;
        in vec2 v_uv_local;

        uniform sampler2D u_velocity;
        uniform vec2 u_vector;

        out vec2 out_velocity;

        void main() {
            vec2 radialVec = (v_uv_local * 2.0 - 1.0);
            float radiusSq = dot(radialVec, radialVec);
            vec2 velocity = texture(u_velocity, v_uv).xy + (1.0 - radiusSq) * u_vector * ${TOUCH_FORCE_SCALE.toFixed(1)};
            float velocityMag = length(velocity);
            out_velocity = velocity / velocityMag * min(velocityMag, ${MAX_VELOCITY.toFixed(1)});
        }`,
        uniforms: [
            { name: 'u_velocity', value: 0, type: INT },
            { name: 'u_vector', value: [0, 0], type: FLOAT },
        ],
    });

    // Touch event handling. Listen on the whole window (not just the canvas)
    // so the fluid can be stirred from anywhere over the article, and skip
    // events that land on the controls panel so sliders do not fling it.
    const activeTouches = {};

    function overControls(e) {
        return e.target && e.target.closest && e.target.closest('#controls-container');
    }

    function onPointerMove(e) {
        if (overControls(e)) return;
        const x = e.clientX;
        const y = e.clientY;

        if (activeTouches[e.pointerId] === undefined) {
            activeTouches[e.pointerId] = { current: [x, y] };
            return;
        }

        activeTouches[e.pointerId].last = activeTouches[e.pointerId].current;
        activeTouches[e.pointerId].current = [x, y];

        const { current, last } = activeTouches[e.pointerId];
        if (!last || (current[0] === last[0] && current[1] === last[1])) return;

        const forceVec = [
            (current[0] - last[0]) * PARAMS.forceScale,
            -(current[1] - last[1]) * PARAMS.forceScale
        ];
        touch.setUniform('u_vector', forceVec);
        composer.stepSegment({
            program: touch,
            input: velocityState,
            output: velocityState,
            position1: [current[0], canvas.height - current[1]],
            position2: [last[0], canvas.height - last[1]],
            thickness: PARAMS.forceRadius,
            endCaps: true,
        });

        // In Suminagashi mode, moving across the screen also drops ink.
        if (PARAMS.render === 'Suminagashi') {
            composer.stepCircle({
                program: dropInk,
                input: colorState,
                output: colorState,
                position: [x, canvas.height - y],
                diameter: 35,
            });
        }
    }

    function onPointerStop(e) {
        delete activeTouches[e.pointerId];
    }

    window.addEventListener('pointermove', onPointerMove, { passive: true });
    window.addEventListener('pointerup', onPointerStop);
    window.addEventListener('pointercancel', onPointerStop);

    // Simulation controls
    controlsPane.addInput(PARAMS, 'trailLength', {
        min: 0,
        max: 100,
        step: 1,
        label: 'Trail Length'
    }).on('change', () => {
        fadeTrails.setUniform('u_increment', -1 / PARAMS.trailLength);
    });

    controlsPane.addInput(PARAMS, 'render', {
        options: {
            Fluid: 'Fluid',
            Suminagashi: 'Suminagashi',
            Pressure: 'Pressure',
            Velocity: 'Velocity',
        },
        label: 'Render Mode',
    });

    // Fluid Dynamics folder
    const fluidFolder = controlsPane.addFolder({ title: 'Fluid Dynamics', expanded: false });

    fluidFolder.addInput(PARAMS, 'viscosity', {
        min: 0.85,
        max: 1.0,
        step: 0.01,
        label: 'ν (viscosity)',
    });

    fluidFolder.addInput(PARAMS, 'forceScale', {
        min: 0.5,
        max: 5.0,
        step: 0.1,
        label: 'F (force)',
    });

    fluidFolder.addInput(PARAMS, 'forceRadius', {
        min: 10,
        max: 80,
        step: 5,
        label: 'r (radius)',
    });

    fluidFolder.addInput(PARAMS, 'jacobiIterations', {
        min: 1,
        max: 20,
        step: 1,
        label: 'n (Jacobi iter)',
    });

    fluidFolder.addInput(PARAMS, 'maxVelocity', {
        min: 10,
        max: 100,
        step: 5,
        label: 'v_max',
    });

    // Particles folder
    const particleFolder = controlsPane.addFolder({ title: 'Particles', expanded: false });

    particleFolder.addInput(PARAMS, 'particleDensity', {
        min: 0.02,
        max: 0.3,
        step: 0.02,
        label: 'ρ (density)',
    });

    particleFolder.addInput(PARAMS, 'particleLifetime', {
        min: 200,
        max: 3000,
        step: 100,
        label: 'τ (lifetime)',
    });

    particleFolder.addInput(PARAMS, 'numRenderSteps', {
        min: 1,
        max: 8,
        step: 1,
        label: 'k (substeps)',
    });

    controlsPane.addButton({ title: 'Reset Simulation' }).on('click', onResize);

    // Info folder
    const simInfoFolder = controlsPane.addFolder({ title: 'Performance', expanded: false });
    const fpsMonitor = { fps: 0 };
    simInfoFolder.addMonitor(fpsMonitor, 'fps', { label: 'FPS' });

    // Resize handler
    function onResize() {
        const { width, height } = getContainerSize();

        composer.resize([width, height]);

        const velocityDimensions = [
            Math.ceil(width / VELOCITY_SCALE_FACTOR),
            Math.ceil(height / VELOCITY_SCALE_FACTOR)
        ];
        velocityState.resize(velocityDimensions);
        divergenceState.resize(velocityDimensions);
        pressureState.resize(velocityDimensions);
        trailState.resize([width, height]);
        colorState.resize([width, height]);
        // Initialize with textured background
        composer.step({
            program: initMarbleTexture,
            output: colorState,
        });

        advection.setUniform('u_dimensions', [width, height]);
        advectParticles.setUniform('u_dimensions', [width, height]);
        advectColors.setUniform('u_dimensions', [width, height]);

        const velocityPxSize = [1 / velocityDimensions[0], 1 / velocityDimensions[1]];
        divergence2D.setUniform('u_pxSize', velocityPxSize);
        jacobi.setUniform('u_pxSize', velocityPxSize);
        gradientSubtraction.setUniform('u_pxSize', velocityPxSize);

        NUM_PARTICLES = calcNumParticles(width, height);

        const positions = new Float32Array(NUM_PARTICLES * 4);
        for (let i = 0; i < positions.length / 4; i++) {
            positions[POSITION_NUM_COMPONENTS * i] = Math.random() * width;
            positions[POSITION_NUM_COMPONENTS * i + 1] = Math.random() * height;
        }
        particlePositionState.resize(NUM_PARTICLES, positions);
        particleInitialState.resize(NUM_PARTICLES, positions);

        const ages = new Int16Array(NUM_PARTICLES);
        for (let i = 0; i < NUM_PARTICLES; i++) {
            ages[i] = Math.round(Math.random() * PARTICLE_LIFETIME);
        }
        particleAgeState.resize(NUM_PARTICLES, ages);
    }

    // Main render loop
    function loop() {
        composer.step({
            program: advection,
            input: [velocityState, velocityState],
            output: velocityState,
        });

        composer.step({
            program: divergence2D,
            input: velocityState,
            output: divergenceState,
        });

        for (let i = 0; i < PARAMS.jacobiIterations; i++) {
            composer.step({
                program: jacobi,
                input: [pressureState, divergenceState],
                output: pressureState,
            });
        }

        composer.step({
            program: gradientSubtraction,
            input: [pressureState, velocityState],
            output: velocityState,
        });

        // Apply viscosity/damping
        applyViscosity.setUniform('u_decay', PARAMS.viscosity);
        applyViscosity.setUniform('u_maxVelocity', PARAMS.maxVelocity);
        composer.step({
            program: applyViscosity,
            input: velocityState,
            output: velocityState,
        });

        if (PARAMS.render === 'Pressure') {
            composer.step({
                program: renderPressure,
                input: pressureState,
            });
        } else if (PARAMS.render === 'Velocity') {
            composer.drawLayerAsVectorField({
                layer: velocityState,
                vectorSpacing: 10,
                vectorScale: 2.5,
                color: [0, 0, 0],
            });
        } else if (PARAMS.render === 'Suminagashi') {
            // Advect colors with velocity field
            composer.step({
                program: advectColors,
                input: [colorState, velocityState],
                output: colorState,
            });

            // Render the suminagashi colors
            composer.step({
                program: renderMarbling,
                input: colorState,
            });
        } else {
            // Fluid mode (particles)
            composer.step({
                program: ageParticles,
                input: particleAgeState,
                output: particleAgeState,
            });

            composer.step({
                program: fadeTrails,
                input: trailState,
                output: trailState,
            });

            for (let i = 0; i < PARAMS.numRenderSteps; i++) {
                composer.step({
                    program: advectParticles,
                    input: [particlePositionState, velocityState, particleAgeState, particleInitialState],
                    output: particlePositionState,
                });

                composer.drawLayerAsPoints({
                    layer: particlePositionState,
                    program: renderParticles,
                    input: [particleAgeState, velocityState],
                    output: trailState,
                    wrapX: true,
                    wrapY: true,
                });
            }

            composer.step({
                program: renderTrails,
                input: trailState,
            });
        }
    }

    // Animation loop
    let time = 0;
    function animate() {
        const { fps, numTicks } = composer.tick();
        time += 0.016; // ~60fps increment

        // Update time uniform for ink color variation
        dropInk.setUniform('u_time', time);

        if (numTicks % 10 === 0) {
            fpsMonitor.fps = fps;
        }

        loop();
        requestAnimationFrame(animate);
    }

    // Initialize
    window.addEventListener('resize', onResize);
    onResize();
    animate();

})();
