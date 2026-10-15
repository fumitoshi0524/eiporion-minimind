// Single-token decoder for the released 8-layer Eiporion checkpoint.
// Projection buffers stay INT8; this browser backend computes in FP32.
export class EiporionGPU {
  async load(url, progress) {
    if (!navigator.gpu) throw new Error('WebGPU is unavailable. Use a recent Chrome or Edge with hardware acceleration enabled.');
    const adapter = await navigator.gpu.requestAdapter({powerPreference: 'high-performance'});
    if (!adapter) throw new Error('No compatible WebGPU adapter was found.');
    this.device = await adapter.requestDevice();
    this.device.lost.then(info => { this.lost = info.message || 'GPU device lost'; });
    this.device.addEventListener('uncapturederror', e => { this.failure = e.error.message; });
    const chunkSize = 4 * 1024 * 1024;
    const response = await fetch(url, {headers:{Range:`bytes=0-${chunkSize-1}`}});
    if (!response.ok) throw new Error(`Weight download failed (${response.status}).`);
    const partial = response.status === 206;
    const size = partial ? Number(response.headers.get('content-range')?.split('/')[1]) : Number(response.headers.get('content-length')) || 78977456;
    const reader = response.body.getReader();
    const chunks = []; let received = 0;
    while (true) {
      const {done, value} = await reader.read(); if (done) break;
      chunks.push(value); received += value.byteLength;
      progress(`Loading weights · ${(received / 1048576).toFixed(1)} / ${(size / 1048576).toFixed(1)} MiB`);
    }
    if(partial){
      for(let start=received;start<size;start+=chunkSize){
        const part=await fetch(url,{headers:{Range:`bytes=${start}-${Math.min(size-1,start+chunkSize-1)}`}});
        if(part.status!==206)throw new Error('The model server did not honor a partial download. Reload and try again.');
        const value=new Uint8Array(await part.arrayBuffer());chunks.push(value);received+=value.length;
        progress(`Loading weights · ${(received / 1048576).toFixed(1)} / ${(size / 1048576).toFixed(1)} MiB`);
      }
    }
    const bytes = new Uint8Array(received); let offset = 0;
    for (const chunk of chunks) {bytes.set(chunk, offset); offset += chunk.length;}
    const headerLength = Number(new DataView(bytes.buffer).getBigUint64(0, true));
    const header = JSON.parse(new TextDecoder().decode(bytes.subarray(8, 8 + headerLength)));
    const base = 8 + headerLength; this.weights = {};
    for (const [name, entry] of Object.entries(header)) {
      if (name === '__metadata__' || name.endsWith('_bit_handle')) continue;
      const [start, end] = entry.data_offsets;
      this.weights[name] = this.buffer(bytes.subarray(base + start, base + end));
    }
    this.embedding = this.weights['model.embed_tokens.weight'] || this.weights['lm_head.weight'];
    if (!this.embedding) throw new Error('The checkpoint has no shared embedding weight.');
    this.params = this.device.createBuffer({size:16, usage:GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST});
    this.scratch = {};
    for (const [name, count] of Object.entries({x:768, b:768, n:768, q:768, k:384, v:384,
        qn:768, kn:384, att:768, o:768, gate:2432, up:2432, act:2432, logits:6400})) {
      this.scratch[name] = this.buffer(new Float32Array(count));
    }
    this.kv = Array.from({length:8}, () => ({k:this.buffer(new Float32Array(512*384)), v:this.buffer(new Float32Array(512*384))}));
    this.readback = this.device.createBuffer({size:6400*4, usage:GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ});
    this.pipelines = new Map(); this.position = 0;
    progress('Preparing GPU kernels…');
    this.device.pushErrorScope('validation');
    await this.build();
    const error = await this.device.popErrorScope();
    if (error) throw new Error(error.message);
    if (this.failure) throw new Error(this.failure);
  }

  buffer(data) {
    const size = Math.max(4, Math.ceil(data.byteLength/4)*4);
    const b = this.device.createBuffer({size, usage:GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC});
    this.device.queue.writeBuffer(b, 0, data); return b;
  }

  async kernel(key, code, buffers, groups) {
    let pipeline = this.pipelines.get(key);
    if (!pipeline) {
      const module = this.device.createShaderModule({code});
      const info = await module.getCompilationInfo();
      const errors = info.messages.filter(m => m.type === 'error');
      if (errors.length) throw new Error(errors.map(m => `${key}: ${m.message}`).join('\n'));
      pipeline = await this.device.createComputePipelineAsync({layout:'auto', compute:{module,entryPoint:'main'}});
      this.pipelines.set(key, pipeline);
    }
    const bind = this.device.createBindGroup({layout:pipeline.getBindGroupLayout(0),
      entries:buffers.map((buffer,binding) => ({binding, resource:{buffer}}))});
    return encoder => {
      const pass = encoder.beginComputePass(); pass.setPipeline(pipeline); pass.setBindGroup(0,bind);
      pass.dispatchWorkgroups(groups); pass.end();
    };
  }

  async matvec(name, input, output, rows, cols, floating=false) {
    const weight = floating ? this.embedding : this.weights[name+'.int_weight'];
    const scales = floating ? null : this.weights[name+'.weight_scale'];
    if (!weight || (!floating && !scales)) throw new Error(`Missing projection: ${name}`);
    const code = `
      @group(0) @binding(0) var<storage,read> w:array<${floating?'f32':'u32'}>;
      @group(0) @binding(1) var<storage,read> x:array<f32>;
      @group(0) @binding(2) var<storage,read_write> y:array<f32>;
      ${floating?'':'@group(0) @binding(3) var<storage,read> scale:array<f32>;'}
      var<workgroup> sums:array<f32,128>;
      @compute @workgroup_size(128) fn main(@builtin(workgroup_id) group:vec3<u32>, @builtin(local_invocation_id) local:vec3<u32>) {
        let row=group.x; let lane=local.x; var sum=0.0;
        for(var j=lane;j<${cols}u;j+=128u) {
          let i=row*${cols}u+j;
          ${floating?'let value=w[i];':'let wordValue=w[i/4u]; let value=f32(i32(((wordValue >> ((i%4u)*8u)) & 255u) << 24u) >> 24);'}
          sum+=value*x[j];
        }
        sums[lane]=sum; workgroupBarrier();
        for(var stride=64u;stride>0u;stride/=2u) {if(lane<stride){sums[lane]+=sums[lane+stride];} workgroupBarrier();}
        if(lane==0u){y[row]=sums[0]${floating?'':'*scale[row]'};}
      }`;
    return this.kernel(`mat-${floating}-${rows}-${cols}`, code, floating?[weight,input,output]:[weight,input,output,scales],rows);
  }

  async norm(input, weight, output, width, heads=1) {
    return this.kernel(`norm-${width}`, `
      @group(0) @binding(0) var<storage,read> x:array<f32>;
      @group(0) @binding(1) var<storage,read> w:array<f32>;
      @group(0) @binding(2) var<storage,read_write> y:array<f32>;
      var<workgroup> sums:array<f32,128>;
      @compute @workgroup_size(128) fn main(@builtin(workgroup_id) g:vec3<u32>,@builtin(local_invocation_id) l:vec3<u32>){
        let lane=l.x; let base=g.x*${width}u; var sum=0.0;
        for(var j=lane;j<${width}u;j+=128u){sum+=x[base+j]*x[base+j];}
        sums[lane]=sum; workgroupBarrier();
        for(var s=64u;s>0u;s/=2u){if(lane<s){sums[lane]+=sums[lane+s];}workgroupBarrier();}
        let inv=inverseSqrt(sums[0]/${width}.0+0.000001);
        for(var j=lane;j<${width}u;j+=128u){y[base+j]=x[base+j]*inv*w[j];}
      }`, [input,weight,output],heads);
  }

  async build() {
    const s=this.scratch, w=this.weights; this.steps=[];
    this.steps.push(await this.kernel('embed',`
      struct Params{position:u32,token:u32,pad0:u32,pad1:u32}
      @group(0) @binding(0) var<storage,read> embedding:array<f32>;
      @group(0) @binding(1) var<storage,read_write> x:array<f32>;
      @group(0) @binding(2) var<uniform> p:Params;
      @compute @workgroup_size(128) fn main(@builtin(global_invocation_id) id:vec3<u32>){
        if(id.x<768u){x[id.x]=embedding[p.token*768u+id.x];}
      }`,[this.embedding,s.x,this.params],6));
    const rope = async (buffer, heads) => this.kernel('rope',`
      struct Params{position:u32,token:u32,pad0:u32,pad1:u32}
      @group(0) @binding(0) var<storage,read_write> x:array<f32>;
      @group(0) @binding(1) var<uniform> p:Params;
      @compute @workgroup_size(48) fn main(@builtin(workgroup_id) g:vec3<u32>,@builtin(local_invocation_id) l:vec3<u32>){
        let i=g.x*96u+l.x; let j=i+48u; let angle=f32(p.position)/pow(1000000.0,f32(l.x)/48.0);
        let a=x[i];let b=x[j]; x[i]=a*cos(angle)-b*sin(angle);x[j]=b*cos(angle)+a*sin(angle);
      }`,[buffer,this.params],heads);
    const add = async (a,b,out) => this.kernel('add',`
      @group(0) @binding(0) var<storage,read> a:array<f32>;
      @group(0) @binding(1) var<storage,read> b:array<f32>;
      @group(0) @binding(2) var<storage,read_write> y:array<f32>;
      @compute @workgroup_size(128) fn main(@builtin(global_invocation_id) id:vec3<u32>){if(id.x<768u){y[id.x]=a[id.x]+b[id.x];}}
    `,[a,b,out],6);
    for(let i=0;i<8;i++) {
      const prefix=`model.layers.${i}`, att=prefix+'.self_attn', mlp=prefix+'.mlp';
      this.steps.push(await this.norm(s.x,w[prefix+'.input_layernorm.weight'],s.n,768));
      this.steps.push(await this.matvec(att+'.q_proj',s.n,s.q,768,768));
      this.steps.push(await this.matvec(att+'.k_proj',s.n,s.k,384,768));
      this.steps.push(await this.matvec(att+'.v_proj',s.n,s.v,384,768));
      this.steps.push(await this.norm(s.q,w[att+'.q_norm.weight'],s.qn,96,8));
      this.steps.push(await this.norm(s.k,w[att+'.k_norm.weight'],s.kn,96,4));
      this.steps.push(await rope(s.qn,8),await rope(s.kn,4));
      this.steps.push(await this.kernel('cache',`
        struct Params{position:u32,token:u32,pad0:u32,pad1:u32}
        @group(0) @binding(0) var<storage,read> k:array<f32>;
        @group(0) @binding(1) var<storage,read> v:array<f32>;
        @group(0) @binding(2) var<storage,read_write> kc:array<f32>;
        @group(0) @binding(3) var<storage,read_write> vc:array<f32>;
        @group(0) @binding(4) var<uniform> p:Params;
        @compute @workgroup_size(128) fn main(@builtin(global_invocation_id) id:vec3<u32>){if(id.x<384u){kc[p.position*384u+id.x]=k[id.x];vc[p.position*384u+id.x]=v[id.x];}}
      `,[s.kn,s.v,this.kv[i].k,this.kv[i].v,this.params],3));
      this.steps.push(await this.kernel('attention',`
        struct Params{position:u32,token:u32,pad0:u32,pad1:u32}
        @group(0) @binding(0) var<storage,read> q:array<f32>;
        @group(0) @binding(1) var<storage,read> k:array<f32>;
        @group(0) @binding(2) var<storage,read> v:array<f32>;
        @group(0) @binding(3) var<storage,read_write> y:array<f32>;
        @group(0) @binding(4) var<uniform> p:Params;
        var<workgroup> scores:array<f32,512>; var<workgroup> reduce:array<f32,128>;
        @compute @workgroup_size(128) fn main(@builtin(workgroup_id) g:vec3<u32>,@builtin(local_invocation_id) l:vec3<u32>){
          let lane=l.x;let head=g.x;let kvhead=head/2u;var maximum=-3.402823e38;
          for(var t=lane;t<=p.position;t+=128u){
            var dot=0.0;for(var d=0u;d<96u;d++){dot+=q[head*96u+d]*k[t*384u+kvhead*96u+d];}
            scores[t]=dot*0.1020620726;maximum=max(maximum,scores[t]);
          }
          reduce[lane]=maximum;workgroupBarrier();
          for(var stride=64u;stride>0u;stride/=2u){if(lane<stride){reduce[lane]=max(reduce[lane],reduce[lane+stride]);}workgroupBarrier();}
          let m=reduce[0];workgroupBarrier();var sum=0.0;
          for(var t=lane;t<=p.position;t+=128u){scores[t]=exp(scores[t]-m);sum+=scores[t];}
          reduce[lane]=sum;workgroupBarrier();
          for(var stride=64u;stride>0u;stride/=2u){if(lane<stride){reduce[lane]+=reduce[lane+stride];}workgroupBarrier();}
          if(lane<96u){var out=0.0;for(var t=0u;t<=p.position;t++){out+=scores[t]/reduce[0]*v[t*384u+kvhead*96u+lane];}y[head*96u+lane]=out;}
        }`,[s.qn,this.kv[i].k,this.kv[i].v,s.att,this.params],8));
      this.steps.push(await this.matvec(att+'.o_proj',s.att,s.o,768,768));
      this.steps.push(await add(s.x,s.o,s.b));
      this.steps.push(await this.norm(s.b,w[prefix+'.post_attention_layernorm.weight'],s.n,768));
      this.steps.push(await this.matvec(mlp+'.gate_proj',s.n,s.gate,2432,768));
      this.steps.push(await this.matvec(mlp+'.up_proj',s.n,s.up,2432,768));
      this.steps.push(await this.kernel('swiglu',`
        @group(0) @binding(0) var<storage,read> gate:array<f32>;
        @group(0) @binding(1) var<storage,read> up:array<f32>;
        @group(0) @binding(2) var<storage,read_write> y:array<f32>;
        @compute @workgroup_size(128) fn main(@builtin(global_invocation_id) id:vec3<u32>){if(id.x<2432u){let g=gate[id.x];y[id.x]=g/(1.0+exp(-g))*up[id.x];}}
      `,[s.gate,s.up,s.act],19));
      this.steps.push(await this.matvec(mlp+'.down_proj',s.act,s.o,768,2432));
      this.steps.push(await add(s.b,s.o,s.x));
    }
    this.steps.push(await this.norm(s.x,w['model.norm.weight'],s.n,768));
    this.steps.push(await this.matvec('',s.n,s.logits,6400,768,true));
  }

  reset(){this.position=0;}
  async token(id) {
    if(this.lost || this.failure) throw new Error(this.lost || this.failure);
    if(this.position>=512) throw new Error('The 512-token context is full. Start a new continuation.');
    this.device.queue.writeBuffer(this.params,0,new Uint32Array([this.position,id,0,0]));
    const encoder=this.device.createCommandEncoder();
    for(const step of this.steps) step(encoder);
    encoder.copyBufferToBuffer(this.scratch.logits,0,this.readback,0,6400*4);
    this.device.queue.submit([encoder.finish()]);
    await this.readback.mapAsync(GPUMapMode.READ);
    const logits=new Float32Array(this.readback.getMappedRange().slice(0));
    this.readback.unmap();this.position++;
    if(this.failure) throw new Error(this.failure);
    return logits;
  }
}
