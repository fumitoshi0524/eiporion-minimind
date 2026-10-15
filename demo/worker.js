import {EiporionGPU} from './webgpu.js';

let model, tokenizer, running=false, cancelled=false;
const send=(type,values={})=>self.postMessage({type,...values});
const status=text=>send('status',{text});

function sample(logits,temperature){
  // Match the Python example's top-k=50 and top-p=0.9 policy.
  const candidates=Array.from(logits,(value,id)=>({id,value:value/temperature}));
  candidates.sort((a,b)=>b.value-a.value);const top=candidates.slice(0,50);
  const max=top[0].value;let sum=0;
  for(const item of top){item.p=Math.exp(item.value-max);sum+=item.p;}
  let accumulated=0, cutoff=top.length;
  for(let i=0;i<top.length;i++){accumulated+=top[i].p/sum;if(accumulated>=0.9){cutoff=i+1;break;}}
  const keep=top.slice(0,cutoff);sum=keep.reduce((s,x)=>s+x.p,0);let r=Math.random()*sum;
  for(const item of keep){r-=item.p;if(r<=0)return item.id;}return keep.at(-1).id;
}

self.onmessage=async({data})=>{
  if(data.type==='stop'){cancelled=true;return;}
  if(running)return;
  running=true;
  try{
    if(data.type==='load'){
      status('Loading tokenizer…');
      const {AutoTokenizer,env}=await import('./vendor/transformers.min.js');
      env.allowRemoteModels=false;env.allowLocalModels=true;
      env.localModelPath=new URL('./',import.meta.url).href;
      tokenizer=await AutoTokenizer.from_pretrained('model',{local_files_only:true});
      model=new EiporionGPU();await model.load(new URL('./model/model.safetensors',import.meta.url),status);
      send('ready');
    }
    if(data.type==='generate'){
      if(!model||!tokenizer)throw new Error('请先加载模型。');
      cancelled=false;model.reset();
      const ids=tokenizer.encode('<|im_start|>'+data.prompt,{add_special_tokens:false});
      if(ids.length>=512)throw new Error('输入过长，请缩短文本开头。总上下文限制为 512 tokens。');
      let logits;
      for(let i=0;i<ids.length;i++){
        if(cancelled)break;
        status(`Reading prompt · ${i+1} / ${ids.length} tokens`);
        logits=await model.token(ids[i]);
      }
      const generated=[],start=performance.now(),limit=Math.min(data.maxTokens,512-ids.length);
      status('正在本机续写…');
      for(let i=0;i<limit&&!cancelled;i++){
        const id=sample(logits,data.temperature);if(id===2)break;
        generated.push(id);send('text',{text:tokenizer.decode(generated,{skip_special_tokens:true}),tokens:generated.length,rate:generated.length/((performance.now()-start)/1000)});
        if(i+1<limit)logits=await model.token(id);
      }
      send('done',{stopped:cancelled});
    }
  }catch(error){send('error',{text:error.message||String(error)});}
  finally{running=false;}
};
