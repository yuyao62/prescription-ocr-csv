/* Whole-object detection; inference and photos stay in the browser.
 * Model source and licensing: see PILL_MODEL.md. */
(() => {
  'use strict';
  const MODEL = 'https://huggingface.co/piky/yolo11/resolve/9ec04d28c48d342906ccaee863a08a6a6394dbc1/yolo11m.onnx';
  const RUNTIME = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.22.0/dist/';
  let sessionPromise;
  async function session() {
    if (!sessionPromise) sessionPromise = (async () => {
      if (!window.ort) await new Promise((resolve,reject) => {
        const script=document.createElement('script');script.src=RUNTIME+'ort.wasm.min.js';
        script.onload=resolve;script.onerror=()=>{script.remove();reject(Error('辨識程式下載失敗，請檢查網路後重試。'));};
        document.head.append(script);
      });
      ort.env.wasm.wasmPaths=RUNTIME;
      ort.env.wasm.numThreads=1;
      return ort.InferenceSession.create(MODEL,{executionProviders:['wasm'],graphOptimizationLevel:'all'});
    })().catch(error=>{sessionPromise=null;throw error;});
    return sessionPromise;
  }
  function suppress(boxes) {
    const selected=[];
    for(const b of boxes.sort((a,b)=>b.score-a.score)) {
      const duplicate=selected.some(q=>{
        const intersection=Math.max(0,Math.min(b.x1,q.x1)-Math.max(b.x0,q.x0))*Math.max(0,Math.min(b.y1,q.y1)-Math.max(b.y0,q.y0));
        const ab=(b.x1-b.x0)*(b.y1-b.y0),aq=(q.x1-q.x0)*(q.y1-q.y0);
        // Remove duplicate classes and a partial box inside the same full object.
        return intersection/Math.max(1e-6,ab+aq-intersection)>.45 || intersection/Math.max(1e-6,Math.min(ab,aq))>.8;
      });
      if(!duplicate)selected.push(b);
    }
    return selected;
  }
  async function detect(source) {
    const net=await session(),side=640,scale=Math.min(side/source.width,side/source.height);
    const w=Math.round(source.width*scale),h=Math.round(source.height*scale);
    const left=Math.floor((side-w)/2),top=Math.floor((side-h)/2);
    const canvas=document.createElement('canvas');canvas.width=canvas.height=side;
    const ctx=canvas.getContext('2d',{willReadFrequently:true});ctx.fillStyle='rgb(114,114,114)';ctx.fillRect(0,0,side,side);
    ctx.drawImage(source,left,top,w,h);
    const pixels=ctx.getImageData(0,0,side,side).data,plane=side*side,input=new Float32Array(3*plane);
    for(let p=0;p<plane;p++){input[p]=pixels[p*4]/255;input[plane+p]=pixels[p*4+1]/255;input[2*plane+p]=pixels[p*4+2]/255;}
    const tensor=new ort.Tensor('float32',input,[1,3,side,side]);let output;
    try {
      output=await net.run({[net.inputNames[0]]:tensor});
      const result=output[net.outputNames[0]],data=result.data,n=result.dims[2],channels=result.dims[1];
      if(channels!==8)throw Error('模型輸出格式不符，請重新整理網頁。');
      const boxes=[];
      for(let i=0;i<n;i++) {
        let cls=0,score=data[4*n+i];
        for(let c=1;c<4;c++)if(data[(4+c)*n+i]>score){score=data[(4+c)*n+i];cls=c;}
        // Model labels: capsule, damaged-pill, foreign-matter, tablet.
        if((cls!==0 && cls!==3)||score<.25)continue;
        const x=data[i],y=data[n+i],bw=data[2*n+i],bh=data[3*n+i];
        const x0=Math.max(0,(x-bw/2-left)/scale),y0=Math.max(0,(y-bh/2-top)/scale);
        const x1=Math.min(source.width,(x+bw/2-left)/scale),y1=Math.min(source.height,(y+bh/2-top)/scale);
        if(![x0,y0,x1,y1,score].every(Number.isFinite)||x1<=x0||y1<=y0)continue;
        boxes.push({x0,y0,x1,y1,score,kind:cls===0?'capsule':'tablet'});
      }
      return suppress(boxes).map(b=>({...b,x:(b.x0+b.x1)/2,y:(b.y0+b.y1)/2,uncertain:b.score<.5,area:(b.x1-b.x0)*(b.y1-b.y0)}));
    } finally {
      tensor.dispose();if(output)for(const value of Object.values(output))value.dispose();
    }
  }
  // A stale photo's result is discarded by the UI; serialize inference so
  // repeated photo/mode changes never run the same WASM session concurrently.
  let pending=Promise.resolve();
  window.PillObjectDetector={detect(source){const run=pending.then(()=>detect(source));pending=run.catch(()=>{});return run;}};
})();
