const $ = id => document.getElementById(id);
const worker = new Worker(new URL('./worker.js',import.meta.url), {type:'module'});
let loaded=false, busy=false;
function controls(){ $('load').disabled=loaded||busy; $('generate').disabled=!loaded||busy; $('stop').disabled=!busy||!loaded; $('prompt').disabled=busy; }
worker.onmessage=({data})=>{
  if(data.type==='status') $('status').textContent=data.text;
  if(data.type==='ready'){loaded=true;busy=false;$('status').textContent='已就绪 · 在你的设备上运行';controls();}
  if(data.type==='text') {$('output').textContent=data.text; $('stats').textContent=`${data.tokens} tokens · ${data.rate.toFixed(1)} tokens/s`;}
  if(data.type==='done'){busy=false;controls();$('status').textContent=data.stopped?'已停止 · 可以重新续写':'已完成 · 可以重新续写';}
  if(data.type==='error'){busy=false;controls();$('error').textContent=data.text;$('status').textContent='暂时无法生成';}
};
worker.onerror=e=>{busy=false;controls();$('error').textContent='运行环境加载失败。请检查网络连接，并使用启用硬件加速的新版 Chrome / Edge。';console.error(e);};
$('load').onclick=()=>{busy=true;$('error').textContent='';controls();worker.postMessage({type:'load'});};
$('generate').onclick=()=>{
  const prompt=$('prompt').value.trim();if(!prompt)return;
  busy=true;controls();$('error').textContent='';$('empty').hidden=true;$('prefix').textContent=prompt;$('output').textContent='';$('stats').textContent='';
  worker.postMessage({type:'generate',prompt,maxTokens:Number($('length').value),temperature:Number($('temperature').value)});
};
$('stop').onclick=()=>worker.postMessage({type:'stop'});
$('temperature').oninput=()=>{$('tempValue').textContent=$('temperature').value;};
document.querySelectorAll('[data-prompt]').forEach(b=>b.onclick=()=>{if(!busy)$('prompt').value=b.dataset.prompt;});
