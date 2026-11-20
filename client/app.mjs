import {validateResult} from './component/index.js';
const drawing=document.querySelector('tamil-symbol-input');
const metadata=document.querySelector('#metadata');
drawing.recognize=async(image,signal)=>{
  const form=new FormData();
  form.append('image',image,'drawing.png');
  const response=await fetch('/api/tamil-symbols/recognize',{
    method:'POST',body:form,signal,credentials:'same-origin',headers:{Accept:'application/json'}
  });
  if(!response.ok)throw new Error('Recognition unavailable. Please try again.');
  return validateResult(await response.json());
};
drawing.addEventListener('recognition-started',()=>{metadata.textContent='';});
drawing.addEventListener('recognition-result',event=>{
  const result=event.detail;
  metadata.textContent=`Model: ${result.model_version}\nProcessing: ${result.processing_ms.toFixed(1)} ms\nRequest: ${result.request_id}`;
});
drawing.addEventListener('recognition-error',()=>{metadata.textContent='';});
