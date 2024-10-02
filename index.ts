
// WebGPU

const canvas = document.createElement('canvas');
document.body.appendChild(canvas);

const adapter = await navigator.gpu.requestAdapter();
const device = await adapter.requestDevice({
    requiredFeatures: ['shader-f16']
});

const UNEXPANDED_CHROMO_STRIDE = 10;
const EXPANDED_CHROMO_STRIDE = 18;

// Each triangle consists of six f32 for vertices and four f32 for colours. The vertices range between -1 and 1.

function fillWithRandomTriangles(arr: Float32Array, triangleAlpha: number) {
    for (let i = 0; i < arr.length; i += UNEXPANDED_CHROMO_STRIDE) {
        for (let j = 0; j < UNEXPANDED_CHROMO_STRIDE - 1; j++)   // initialise colours
            arr[i + j] = Math.random();
        arr[i + UNEXPANDED_CHROMO_STRIDE - 1] = triangleAlpha;
    }
}

// 10 f32 per triangle to 18 f32 per expanded
function expandTriangles(tri: Float32Array, vertices: Float32Array) {
    console.assert(tri.length / 10 === vertices.length / 18);

    for (let i = 0, j = 0; i < tri.length; i += 10, j += 18) {
        for (let k = 0; k < 3; k++) {
            vertices[j + k * 6] = tri[i + k * 2];  // position
            vertices[j + k * 6 + 1] = tri[i + k * 2 + 1];
            vertices[j + k * 6 + 2] = tri[i + 6];  // colour
            vertices[j + k * 6 + 3] = tri[i + 7];
            vertices[j + k * 6 + 4] = tri[i + 8];
            vertices[j + k * 6 + 5] = tri[i + 9];
        }
    }
}

function clamp(x: number, min: number, max: number) {
    return Math.min(Math.max(x, min), max);
}

const PROBABILITY_TO_PERTURB_COORDINATE = 0.25;
const PROBABILITY_TO_PERTURB_COLOUR = 0.25;

function mutateChromo1(tri: Float32Array, perturbation: number) {
    for (let i = 0; i < tri.length; i += UNEXPANDED_CHROMO_STRIDE) {
        for (let j = 0; j < 6; j++) {
            if (Math.random() < PROBABILITY_TO_PERTURB_COORDINATE) {
                tri[i + j] = clamp(tri[i + j] + 2 * (Math.random() - 0.5) * perturbation, 0, 1);
            }
        }

        for (let j = 6; j < 9 /* exclude alpha channel */; j++) {
            if (Math.random() < PROBABILITY_TO_PERTURB_COLOUR) {
                tri[i + j] = clamp(tri[i + j] + (Math.random() - 0.5) * perturbation, 0, 1);
            }
        }
    }
}

function crossover1p(chromo1: Float32Array, chromo2: Float32Array, target: Float32Array, triangleCount: number) {
    const crossoverPoint = Math.floor(Math.random() * triangleCount) * UNEXPANDED_CHROMO_STRIDE;

    for (let i = 0; i < crossoverPoint; i++) {
        target[i] = chromo1[i];
    }

    for (let i = crossoverPoint; i < target.length; i++) {
        target[i] = chromo2[i];
    }
}

function crossoverRandom(chromo1: Float32Array, chromo2: Float32Array, target: Float32Array, triangleCount: number) {
    for (let i = 0; i < triangleCount; i++) {
        const source = Math.random() < 0.5 ? chromo1 : chromo2;
        const sourceOffset = i * UNEXPANDED_CHROMO_STRIDE;
        const targetOffset = i * UNEXPANDED_CHROMO_STRIDE;

        for (let j = 0; j < UNEXPANDED_CHROMO_STRIDE; j++) {
            target[targetOffset + j] = source[sourceOffset + j];
        }
    }
}

const DrawProgram = `
struct ChromoVertex {
    @location(0) position: vec2f,
    @location(1) color: vec4f,
};

struct VsOutput {
  @builtin(position) position: vec4f,
  @location(0) color: vec4f,
};

struct Uniforms {
    m: u32,
    n: u32,
    targetWidth: u32,
    targetHeight: u32,
    trianglesPerChromo: u32,
};

@group(0) @binding(0) var<uniform> uniforms: Uniforms;

@vertex fn vs(
    @builtin(vertex_index) vertexIndex: u32,
    v: ChromoVertex
) -> VsOutput {
    let draw_index = vertexIndex / (3 * uniforms.trianglesPerChromo);
    
    let draw_index_y = draw_index / uniforms.m;
    let draw_index_x = draw_index - draw_index_y * uniforms.m;
    
    let per_inst_width = 2.0 / f32(uniforms.m);
    let per_inst_height = 2.0 / f32(uniforms.n);
    
    let base_x = -1.0 + per_inst_width * f32(draw_index_x);
    let base_y = -1.0 + per_inst_height * f32(draw_index_y);
    
    var pos = v.position;
    pos.x = base_x + pos.x * per_inst_width;
    pos.y = base_y + pos.y * per_inst_height;
    
    return VsOutput(
        vec4f(pos, 0.0, 1.0),
        v.color
    );
}

@fragment fn fs(v: VsOutput) -> @location(0) vec4f {
    return v.color;
}
`;

// Inputs:
//  M, N, targetWidth, targetHeight, imageToApproximate, offset into storage buffer
const FitnessProgram = `
struct Uniforms {
    m: u32,
    n: u32,
    targetWidth: u32,
    targetHeight: u32,
};

@group(0) @binding(0) var<uniform> uniforms: Uniforms;
@group(0) @binding(1) var imageToApproximate: texture_2d<f32>;
@group(0) @binding(2) var rasterised: texture_2d<f32>;
@group(0) @binding(3) var<storage, read_write> fitnesses: array<f32>;

@compute @workgroup_size(1) fn main(
    @builtin(global_invocation_id) globalInvocationID: vec3<u32>
) {
    let x = i32(globalInvocationID.x);
    let y = i32(uniforms.n - globalInvocationID.y - 1);  // flip around
    
    let dx = i32(uniforms.targetWidth);
    let dy = i32(uniforms.targetHeight);
    
    var fitness = 0.0;
    
    for (var i = 0; i < dx; i = i + 1) {
        for (var j = 0; j < dy; j = j + 1) {
            let target_ = textureLoad(imageToApproximate, vec2<i32>(i, j), 0);
            let approx = textureLoad(rasterised, vec2<i32>(x * dx + i, y * dy + j), 0);
            
            let delta = target_ - approx;
            
            fitness += dot(delta, delta);
        }
    }
    
    fitnesses[i32(globalInvocationID.y) * i32(uniforms.m) + x] = fitness;
}
`;

type InstanceBuffers = {
    // struct {
    //    float2 pos;
    //    float4 color;
    // } chromos[POP_SIZE * TRIANGLE_COUNT];
    chromos: GPUBuffer;

    // Uniforms for the draw pass
    // struct {
    //   uint m;
    //   uint n;
    //   uint targetWidth;
    //   uint targetHeight;
    //   uint trianglesPerChromo;
    // } uniforms;
    drawUniforms: GPUBuffer;

    // Uniforms for the MSE compute pass
    // struct {
    //   uint m;
    //   uint n;
    //   uint targetWidth;
    //   uint targetHeight;
    //   uint writeOffset;
    // } uniforms;
    computeUniforms: GPUBuffer;

    // Texture containing the target image we are trying to approximate
    goalImage: GPUTexture;

    // Texture to render to
    renderTarget: GPUTexture;

    // float fitness[POP_SIZE];
    fitness: GPUBuffer;

    fitnessReader: GPUBuffer;
}

type InstanceOptions = {
    triangleAlpha: number;
    popSize: number;
    trianglesPerChromo: number;
    targetImage: ImageData;
    backgroundColor: [number, number, number, number];
    forceBatch?: [number, number];
};

function parseFloat16(u16: number) {
    const sign = (u16 & 0x8000) >> 15;
    const exp = (u16 & 0x7C00) >> 10;
    let mantissa = u16 & 0x3FF;

    if (exp === 0) {
        if (mantissa !== 0) { // denormalised
            mantissa *= 5.960464477539063e-8;
        }
    } else if (exp === 31) {
        mantissa = mantissa === 0 ? Infinity : NaN;
    } else {
        mantissa = (1 << exp) * (0.000030517578125 + mantissa * 2.9802322387695312e-8);
    }

    return sign === 0 ? mantissa : -mantissa;
}

class Instance {
    ctx: CanvasRenderingContext2D;

    buffers: InstanceBuffers;
    generation: Float32Array;
    expanded: Float32Array;

    M: number;
    N: number;

    module: GPUShaderModule;
    pipeline: GPURenderPipeline;
    private targetDims: [number, number];

    async downloadIntermediateTexture(): Promise<ImageData> {
        const texture = this.buffers.renderTarget;
        const { width, height } = texture;

        const SIZEOF_PIXEL = 8; // 4 * sizeof(f16)

        const bytesPerRow = (texture.width * SIZEOF_PIXEL + 255) & ~255;  // round up to multiple of 256
        const buffer = device.createBuffer({
            label: 'download buffer',
            size: bytesPerRow * texture.height * SIZEOF_PIXEL,
            usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
        });

        const encoder = device.createCommandEncoder();
        encoder.copyTextureToBuffer(
            { texture, mipLevel: 0, origin: [0, 0, 0] },
            { buffer, offset: 0, bytesPerRow },
            [width, height]
        );
        device.queue.submit([encoder.finish()]);

        await buffer.mapAsync(GPUMapMode.READ);
        const data = new Uint16Array(buffer.getMappedRange());

        const result = new Uint8ClampedArray(width * height * 4);
        for (let i = 0; i < height; ++i) {  // for each row
            const offs = i * bytesPerRow / SIZEOF_PIXEL;

            for (let j = 0; j < texture.width * 4; ++j) {  // copy row
                result[i * width * 4 + j] = parseFloat16(data[offs * 4 + j]) * 255.0;
            }
        }

        buffer.destroy();
        return new ImageData(result, width, height);
    }

    async showIntermediate() {
        const downloaded = await this.downloadIntermediateTexture();
        const { ctx } = this;
        const { canvas } = ctx;

        const w = canvas.width = downloaded.width;
        const h = canvas.height = downloaded.height;
        const dpr = window.devicePixelRatio;
        canvas.style.width = `${w / dpr}px`;
        canvas.style.height = `${h / dpr}px`;

        ctx.putImageData(downloaded, 0, 0);
    }

    constructor(readonly canvas: HTMLCanvasElement, readonly options: InstanceOptions) {
        this.ctx = canvas.getContext('2d');
        this.buffers = {} as any;

        this.generation = new Float32Array(options.popSize * options.trianglesPerChromo * UNEXPANDED_CHROMO_STRIDE);
        this.expanded = new Float32Array(options.popSize * options.trianglesPerChromo * EXPANDED_CHROMO_STRIDE);

        const img = options.targetImage;

        // Optimise for this to be the side length of the rendered-to texture
        const GOAL_DIM = 2048;
        this.M = options.forceBatch?.[0] ?? Math.floor(GOAL_DIM / img.width);
        this.N = options.forceBatch?.[1] ?? Math.floor(GOAL_DIM / img.height);

        this.targetDims = [ this.M * img.width, this.N * img.height ];

        canvas.width = this.targetDims[0];
        canvas.height = this.targetDims[1];

        fillWithRandomTriangles(this.generation, options.triangleAlpha);
        expandTriangles(this.generation, this.expanded);

        // Format of the intermediate rendered-to texture
        const intermediateFormat: GPUTextureFormat = "rgba16float";

        const module = this.module = device.createShaderModule({
            label: 'draw program',
            code: DrawProgram
        });

        const goalImage = this.buffers.goalImage = device.createTexture({
            size: [img.width, img.height],
            format: intermediateFormat,
            usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST
        });

        device.queue.copyExternalImageToTexture({
            source: img,
            flipY: true,
        }, { texture: goalImage }, [img.width, img.height]);

        this.buffers.renderTarget = device.createTexture({
            size: this.targetDims,
            format: intermediateFormat,
            usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_SRC | GPUTextureUsage.TEXTURE_BINDING
        });

        this.buffers.fitness = device.createBuffer({
            label: 'fitness',
            size: options.popSize * 4,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC
        });

        this.buffers.fitnessReader = device.createBuffer({
            label: 'fitness reader',
            size: options.popSize * 4,
            usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ
        });

        const pipeline = this.pipeline = device.createRenderPipeline({
            label: 'draw pipeline',
            layout: 'auto',
            vertex: {
                entryPoint: 'vs',
                module,
                buffers: [
                    {
                        arrayStride: 6 * 4,
                        attributes: [
                            // position
                            {shaderLocation: 0, offset: 0, format: 'float32x2'},
                            {shaderLocation: 1, offset: 2 * 4, format: 'float32x4'},
                        ],
                    }
                ]
            },
            fragment: {
                entryPoint: 'fs',
                module,
                targets: [{
                    format: intermediateFormat,
                    blend: {
                        color: {
                            srcFactor: 'src-alpha',
                            dstFactor: 'one-minus-src-alpha'
                        },
                        alpha: {
                            srcFactor: 'zero',
                            dstFactor: 'one'
                        },
                    }
                }],
            },
        });

        const renderPassDescriptor: GPURenderPassDescriptor = {
            label: 'our basic canvas renderPass',
            colorAttachments: [
                {
                    // view: <- to be filled out when we render
                    clearValue: options.backgroundColor,
                    loadOp: 'clear',
                    storeOp: 'store',
                    view: this.buffers.renderTarget.createView()
                } as GPURenderPassColorAttachment,
            ],
        };

        // make a command encoder to start encoding commands
        const encoder = device.createCommandEncoder({label: 'our encoder'});

        const chromos = device.createBuffer({
            label: 'chromos',
            size: this.expanded.byteLength,
            usage: GPUBufferUsage.VERTEX,
            mappedAtCreation: true,
        });

        new Float32Array(chromos.getMappedRange()).set(this.expanded);
        chromos.unmap();

        const drawUniforms = device.createBuffer({
            label: 'draw uniforms',
            size: 20,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });

        const bindGroup = device.createBindGroup({
            label: 'draw bind group',
            layout: pipeline.getBindGroupLayout(0),
            entries: [
                { binding: 0, resource: { buffer: drawUniforms }},
            ],
        });

        device.queue.writeBuffer(
            drawUniforms,
            0,
            new Uint32Array([
                this.M, this.N,
                img.width, img.height,
                options.trianglesPerChromo
            ])
        );

        // make a render pass encoder to encode render specific commands
        const pass = encoder.beginRenderPass(renderPassDescriptor);
        pass.setPipeline(pipeline);
        pass.setVertexBuffer(0, chromos);
        pass.setBindGroup(0, bindGroup);
        pass.draw(options.popSize * options.trianglesPerChromo * 3);  // call our vertex shader 3 times
        pass.end();

        const computePipeline = device.createComputePipeline({
            label: 'fitness',
            compute: {
                module: device.createShaderModule({
                    label: 'fitness program',
                    code: FitnessProgram
                }),
                entryPoint: 'main',
            },
            layout: "auto",
        });

        this.buffers.chromos = chromos;
        this.buffers.drawUniforms = drawUniforms;

        const computeUniforms = device.createBuffer({
            label: 'compute uniforms',
            size: 16,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });
        this.buffers.computeUniforms = computeUniforms;

        const computeBindGroup = device.createBindGroup({
            label: 'compute bind group',
            layout: computePipeline.getBindGroupLayout(0),
            entries: [
                { binding: 0, resource: { buffer: this.buffers.computeUniforms }},
                { binding: 1, resource: this.buffers.goalImage.createView() },
                { binding: 2, resource: this.buffers.renderTarget.createView() },
                { binding: 3, resource: { buffer: this.buffers.fitness, offset: 0 }},
            ],
        });

        const computePass = encoder.beginComputePass({ label: "compute" });
        computePass.setPipeline(computePipeline);
        computePass.setBindGroup(0, computeBindGroup);

        device.queue.writeBuffer(
            computeUniforms,
            0,
            new Uint32Array([
                this.M, this.N,
                img.width, img.height
            ])
        );

        computePass.dispatchWorkgroups(this.M, this.N);
        computePass.end();

        encoder.copyBufferToBuffer(this.buffers.fitness, 0, this.buffers.fitnessReader, 0, options.popSize * 4);

        const commandBuffer = encoder.finish();
        device.queue.submit([commandBuffer]);

        (async() => {
            const f = this.buffers.fitnessReader;
            await f.mapAsync(GPUMapMode.READ);
            const fitnesses = new Float32Array(f.getMappedRange()).slice();

            console.log({ fitnesses });

            f.unmap();
        })();


        this.showIntermediate();
    }
}

// Load image data from pipe.jpeg
const img = new Promise<ImageData>((resolve, reject) => {
    const img = new Image();
    img.onload = () => {
        const canvas = document.createElement('canvas');
        canvas.width = img.width;
        canvas.height = img.height;
        const ctx = canvas.getContext('2d');
        ctx.drawImage(img, 0, 0);
        resolve(ctx.getImageData(0, 0, img.width, img.height));
    };
    img.src = 'pipe.jpeg';
});

new Instance(canvas, {
    triangleAlpha: 0.35,
    popSize: 50,
    trianglesPerChromo: 50,
    targetImage: await img,
    backgroundColor: [0.5, 0.5, 0.5, 1.0]
});
