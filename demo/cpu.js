// Compatibility backend: original INT8 projections, no WebGPU dependency.
// Runs in the model worker so matrix operations do not block the page.
import {downloadCheckpoint} from './weights.js';
export class EiporionCPU {
  async load(url, progress) {
    progress('Loading weights for CPU…');
    const bytes=await downloadCheckpoint(url,progress);
    const length=Number(new DataView(bytes.buffer).getBigUint64(0,true));
    const header=JSON.parse(new TextDecoder().decode(bytes.subarray(8,8+length)));
    this.weights={};
    for(const [name,entry] of Object.entries(header)){
      if(name==='__metadata__'||name.endsWith('_bit_handle'))continue;
      const [start,end]=entry.data_offsets, begin=8+length+start;
      if(entry.dtype==='I8')this.weights[name]=new Int8Array(bytes.buffer,begin,end-start);
      else if(entry.dtype==='F32')this.weights[name]=new Float32Array(bytes.buffer,begin,(end-start)/4);
      else throw new Error(`Unsupported tensor dtype: ${entry.dtype}`);
    }
    this.embedding=this.weights['model.embed_tokens.weight']||this.weights['lm_head.weight'];
    if(!this.embedding)throw new Error('Missing shared embedding weight.');
    this.s={};
    for(const [name,count] of Object.entries({x:768,b:768,n:768,q:768,k:384,v:384,qn:768,kn:384,att:768,o:768,gate:2432,up:2432,act:2432,logits:6400}))this.s[name]=new Float32Array(count);
    this.kv=Array.from({length:8},()=>({k:new Float32Array(512*384),v:new Float32Array(512*384)}));
    this.scores=new Float64Array(512);this.position=0;
  }
  matvec(name,x,y,rows,cols){
    const w=this.weights[name+'.int_weight'],scale=this.weights[name+'.weight_scale'];
    for(let row=0;row<rows;row++){
      const base=row*cols;let a=0,b=0,c=0,d=0;
      for(let j=0;j<cols;j+=4){a+=w[base+j]*x[j];b+=w[base+j+1]*x[j+1];c+=w[base+j+2]*x[j+2];d+=w[base+j+3]*x[j+3];}
      y[row]=(a+b+c+d)*scale[row];
    }
  }
  norm(x,w,y,width,heads=1){
    for(let head=0;head<heads;head++){
      const base=head*width;let sum=0;for(let j=0;j<width;j++)sum+=x[base+j]*x[base+j];
      const inv=1/Math.sqrt(sum/width+1e-6);for(let j=0;j<width;j++)y[base+j]=x[base+j]*inv*w[j];
    }
  }
  rope(x,heads){
    for(let head=0;head<heads;head++)for(let d=0;d<48;d++){
      const i=head*96+d,j=i+48,angle=this.position/Math.pow(1e6,d/48),a=x[i],b=x[j];
      x[i]=a*Math.cos(angle)-b*Math.sin(angle);x[j]=b*Math.cos(angle)+a*Math.sin(angle);
    }
  }
  reset(){this.position=0;}
  async token(id){
    if(this.position>=512)throw new Error('The 512-token context is full.');
    const s=this.s,w=this.weights,pos=this.position;
    s.x.set(this.embedding.subarray(id*768,(id+1)*768));
    for(let layer=0;layer<8;layer++){
      const prefix=`model.layers.${layer}`,att=prefix+'.self_attn',mlp=prefix+'.mlp',kv=this.kv[layer];
      this.norm(s.x,w[prefix+'.input_layernorm.weight'],s.n,768);
      this.matvec(att+'.q_proj',s.n,s.q,768,768);this.matvec(att+'.k_proj',s.n,s.k,384,768);this.matvec(att+'.v_proj',s.n,s.v,384,768);
      this.norm(s.q,w[att+'.q_norm.weight'],s.qn,96,8);this.norm(s.k,w[att+'.k_norm.weight'],s.kn,96,4);
      this.rope(s.qn,8);this.rope(s.kn,4);kv.k.set(s.kn,pos*384);kv.v.set(s.v,pos*384);
      for(let head=0;head<8;head++){
        const qbase=head*96,kbase=Math.floor(head/2)*96;let maximum=-Infinity;
        for(let t=0;t<=pos;t++){
          let dot=0;for(let d=0;d<96;d++)dot+=s.qn[qbase+d]*kv.k[t*384+kbase+d];
          this.scores[t]=dot/Math.sqrt(96);maximum=Math.max(maximum,this.scores[t]);
        }
        let sum=0;for(let t=0;t<=pos;t++){this.scores[t]=Math.exp(this.scores[t]-maximum);sum+=this.scores[t];}
        for(let d=0;d<96;d++){let value=0;for(let t=0;t<=pos;t++)value+=this.scores[t]/sum*kv.v[t*384+kbase+d];s.att[qbase+d]=value;}
      }
      this.matvec(att+'.o_proj',s.att,s.o,768,768);for(let j=0;j<768;j++)s.b[j]=s.x[j]+s.o[j];
      this.norm(s.b,w[prefix+'.post_attention_layernorm.weight'],s.n,768);
      this.matvec(mlp+'.gate_proj',s.n,s.gate,2432,768);this.matvec(mlp+'.up_proj',s.n,s.up,2432,768);
      for(let j=0;j<2432;j++){const g=s.gate[j];s.act[j]=g/(1+Math.exp(-g))*s.up[j];}
      this.matvec(mlp+'.down_proj',s.act,s.o,768,2432);for(let j=0;j<768;j++)s.x[j]=s.b[j]+s.o[j];
    }
    this.norm(s.x,w['model.norm.weight'],s.n,768);
    for(let row=0;row<6400;row++){
      let sum=0;const base=row*768;for(let j=0;j<768;j++)sum+=this.embedding[base+j]*s.n[j];s.logits[row]=sum;
    }
    this.position++;
    await new Promise(resolve=>setTimeout(resolve,0));
    return s.logits;
  }
}
