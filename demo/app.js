const $ = id => document.getElementById(id);
const english=new URLSearchParams(location.search).get('lang')==='en';
const local=(en,zh)=>english?en:zh;
document.documentElement.lang=english?'en':'zh-CN';
const languageLink=document.createElement('a');languageLink.href=english?'?lang=zh':'?lang=en';languageLink.textContent=english?'中文':'English';
document.querySelector('nav').insertBefore(languageLink,document.querySelector('nav a'));
if(english){
  document.querySelector('header p').textContent='Write an opening line and see what comes next. Download the weights once and generate text on your own device.';
  $('status').textContent='Not loaded · WebGPU with automatic CPU fallback';
  $('load').textContent='Load model';
  document.querySelector('label[for="prompt"]').textContent='PROMPT';
  $('prompt').value='Artificial intelligence is';
  const prompts=['Artificial intelligence is','On a quiet evening,','Learning a new language'];
  document.querySelectorAll('[data-prompt]').forEach((b,i)=>{b.dataset.prompt=prompts[i];b.textContent=prompts[i];});
  document.querySelector('.editor small').textContent='A pretrained model for text completion; not instruction-tuned for chat.';
  document.querySelector('label[for="length"]').firstChild.textContent='New tokens ';
  document.querySelector('label[for="temperature"]').firstChild.textContent='Temperature ';
  $('stop').textContent='Stop';$('generate').textContent='Continue';
  document.querySelector('.result h2').textContent='CONTINUATION';$('empty').textContent='Your continuation will appear here.';
  document.querySelector('footer span').textContent='Local inference · WebGPU / CPU · INT8 projection storage';
  document.querySelector('footer a').textContent='Read the research report ↗';
  document.querySelector('footer a').href='https://github.com/fumitoshi0524/eiporion-minimind/blob/main/README.md';
}
const worker = new Worker(new URL('./worker.js?v=2',import.meta.url), {type:'module'});
let loaded=false, busy=false, backend='WebGPU';
function controls(){ $('load').disabled=loaded||busy; $('generate').disabled=!loaded||busy; $('stop').disabled=!busy||!loaded; $('prompt').disabled=busy; }
worker.onmessage=({data})=>{
  if(data.type==='status') $('status').textContent=data.text;
  if(data.type==='ready'){loaded=true;busy=false;backend=data.backend;$('status').textContent=backend==='CPU'?local('Ready · CPU compatibility mode (slower)','已就绪 · CPU 兼容模式（速度较慢）'):local('Ready · WebGPU on your device','已就绪 · WebGPU 本机运行');controls();}
  if(data.type==='text') {$('output').textContent=data.text; $('stats').textContent=`${data.tokens} tokens · ${data.rate.toFixed(1)} tokens/s`;}
  if(data.type==='done'){busy=false;controls();$('status').textContent=(data.stopped?local('Stopped','已停止'):local('Completed','已完成'))+' · '+backend;}
  if(data.type==='error'){busy=false;controls();$('error').textContent=data.text;$('status').textContent=local('Unable to generate','暂时无法生成');}
};
worker.onerror=e=>{busy=false;controls();$('error').textContent=local('The runtime could not load. Check your connection and reload the page.','运行环境加载失败。请检查网络连接并刷新页面。');console.error(e);};
$('load').onclick=()=>{busy=true;$('error').textContent='';controls();worker.postMessage({type:'load',lang:english?'en':'zh'});};
$('generate').onclick=()=>{
  const prompt=$('prompt').value.trim();if(!prompt)return;
  busy=true;controls();$('error').textContent='';$('empty').hidden=true;$('prefix').textContent=prompt;$('output').textContent='';$('stats').textContent='';
  worker.postMessage({type:'generate',prompt,maxTokens:Number($('length').value),temperature:Number($('temperature').value)});
};
$('stop').onclick=()=>worker.postMessage({type:'stop'});
$('temperature').oninput=()=>{$('tempValue').textContent=$('temperature').value;};
document.querySelectorAll('[data-prompt]').forEach(b=>b.onclick=()=>{if(!busy)$('prompt').value=b.dataset.prompt;});
