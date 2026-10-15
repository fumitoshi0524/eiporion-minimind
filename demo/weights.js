export async function downloadCheckpoint(url, progress) {
  const chunkSize=4*1024*1024;
  const response=await fetch(url,{headers:{Range:`bytes=0-${chunkSize-1}`}});
  if(!response.ok)throw new Error(`Weight download failed (${response.status}).`);
  const partial=response.status===206;
  const total=partial?Number(response.headers.get('content-range')?.split('/')[1]):Number(response.headers.get('content-length'))||78977456;
  const reader=response.body.getReader(),chunks=[];let received=0;
  const report=()=>progress(`${(received/1048576).toFixed(1)} / ${(total/1048576).toFixed(1)} MiB`);
  while(true){const {done,value}=await reader.read();if(done)break;chunks.push(value);received+=value.length;report();}
  if(partial)for(let start=received;start<total;start+=chunkSize){
    const part=await fetch(url,{headers:{Range:`bytes=${start}-${Math.min(total-1,start+chunkSize-1)}`}});
    if(part.status!==206)throw new Error('Model download interrupted. Please reload and retry.');
    const value=new Uint8Array(await part.arrayBuffer());chunks.push(value);received+=value.length;report();
  }
  const bytes=new Uint8Array(received);let offset=0;
  for(const chunk of chunks){bytes.set(chunk,offset);offset+=chunk.length;}
  return bytes;
}
