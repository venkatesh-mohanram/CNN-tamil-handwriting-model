import {RecognitionController,                                      } from './recognition.js';
export {validateResult, TAMIL_SYMBOLS} from './recognition.js';
                                                                  

                               
/** Framework-independent input. The host owns transport and authentication. */
export class TamilSymbolInput extends HTMLElement {
          callback                       ;
          canvas = document.createElement('canvas');
          status = document.createElement('p');
          clearButton = document.createElement('button');
          undoButton = document.createElement('button');
          submitButton = document.createElement('button');
          strokes            = [];
          activePointer                = null;
          controller = new RecognitionController(event=>this.onRecognition(event));
  constructor() {
    super();
    const shadow=this.attachShadow({mode:'open'});
    const style=document.createElement('style');
    style.textContent=':host{display:block;max-width:24rem;font:1rem system-ui,sans-serif}canvas{display:block;width:100%;aspect-ratio:1;touch-action:none;border:2px solid #475569;border-radius:.5rem;background:#fff}button{font:inherit;padding:.65rem 1rem;margin:.5rem .4rem .25rem 0;min-height:44px}button:focus-visible{outline:3px solid #2563eb;outline-offset:2px}p{min-height:1.5em}';
    const instruction=document.createElement('p');
    instruction.textContent='Draw one Tamil vowel or ஃ, then select Recognize.';
    this.canvas.setAttribute('aria-label','Tamil character drawing area. Use a mouse, touch, or stylus.');
    this.canvas.setAttribute('role','img');
    this.clearButton.textContent='Clear';this.undoButton.textContent='Undo';this.submitButton.textContent='Recognize';
    for(const button of [this.clearButton,this.undoButton,this.submitButton])button.type='button';
    this.status.setAttribute('role','status');this.status.setAttribute('aria-live','polite');
    this.status.setAttribute('aria-atomic','true');
    shadow.append(style,instruction,this.canvas,this.clearButton,this.undoButton,this.submitButton,this.status);
    this.clearButton.addEventListener('click',()=>this.clear());
    this.undoButton.addEventListener('click',()=>this.undo());
    this.submitButton.addEventListener('click',()=>void this.submit());
    this.canvas.addEventListener('pointerdown',event=>this.startStroke(event));
    this.canvas.addEventListener('pointermove',event=>this.extendStroke(event));
    this.canvas.addEventListener('pointerup',event=>this.finishStroke(event));
    this.canvas.addEventListener('pointercancel',event=>this.finishStroke(event));
    this.canvas.addEventListener('lostpointercapture',()=>{this.activePointer=null;this.updateControls();});
  }
  get recognize()                        {return this.callback;}
  set recognize(callback                       ) {this.callback=callback;this.updateControls();}
  connectedCallback()       {
    const ratio=Math.max(1,Math.min(window.devicePixelRatio||1,3));
    this.canvas.width=Math.round(320*ratio);this.canvas.height=Math.round(320*ratio);
    this.renderDrawing();this.updateControls();
  }
  disconnectedCallback()       {this.controller.cancel();this.activePointer=null;this.strokes=[];this.renderDrawing();}
  clear()       {
    if(this.controller.pending)return;
    this.strokes=[];this.activePointer=null;this.status.textContent='Drawing cleared.';
    this.renderDrawing();this.updateControls();
  }
  undo()       {
    if(this.controller.pending || this.activePointer!==null)return;
    this.strokes.pop();this.status.textContent='Last stroke removed.';
    this.renderDrawing();this.updateControls();
  }
          point(event              )        {
    const rect=this.canvas.getBoundingClientRect();
    return {x:Math.max(0,Math.min(1,(event.clientX-rect.left)/rect.width)),y:Math.max(0,Math.min(1,(event.clientY-rect.top)/rect.height))};
  }
          startStroke(event              )       {
    if(this.controller.pending || this.activePointer!==null || !event.isPrimary || event.button!==0)return;
    event.preventDefault();this.activePointer=event.pointerId;
    this.canvas.setPointerCapture(event.pointerId);this.strokes.push([this.point(event)]);
    this.status.textContent='';this.renderDrawing();this.updateControls();
  }
          extendStroke(event              )       {
    if(event.pointerId!==this.activePointer)return;
    event.preventDefault();this.strokes[this.strokes.length-1] .push(this.point(event));this.renderDrawing();
  }
          finishStroke(event              )       {
    if(event.pointerId!==this.activePointer)return;
    this.activePointer=null;
    if(this.canvas.hasPointerCapture(event.pointerId))this.canvas.releasePointerCapture(event.pointerId);
    this.updateControls();
  }
          paint(canvas                   )       {
    const ctx=canvas.getContext('2d');if(!ctx)throw new Error('Canvas is unavailable');
    ctx.fillStyle='white';ctx.fillRect(0,0,canvas.width,canvas.height);
    ctx.strokeStyle='black';ctx.lineWidth=canvas.width*.025;ctx.lineCap='round';ctx.lineJoin='round';
    for(const stroke of this.strokes){
      const first=stroke[0];if(!first)continue;
      ctx.beginPath();ctx.moveTo(first.x*canvas.width,first.y*canvas.height);
      // A dot is a real stroke, particularly for aytham; never discard it.
      if(stroke.length===1)ctx.lineTo(first.x*canvas.width+.01,first.y*canvas.height+.01);
      else for(const point of stroke.slice(1))ctx.lineTo(point.x*canvas.width,point.y*canvas.height);
      ctx.stroke();
    }
  }
          renderDrawing()       {this.paint(this.canvas);}
          updateControls()       {
    const pending=this.controller.pending;const empty=this.strokes.length===0;
    this.clearButton.disabled=pending||empty;this.undoButton.disabled=pending||empty||this.activePointer!==null;
    this.submitButton.disabled=pending||empty||this.activePointer!==null||!this.callback;
    this.canvas.setAttribute('aria-busy',String(pending));
  }
          async submit()                {
    if(this.controller.pending||!this.callback||this.strokes.length===0||this.activePointer!==null)return;
    const callback=this.callback;
    // Reserve the request before asynchronous PNG conversion to prevent double clicks.
    await this.controller.submit(new Blob(),async(_unused,signal)=>{
      const output=document.createElement('canvas');output.width=320;output.height=320;this.paint(output);
      const image=await new Promise      ((resolve,reject)=>output.toBlob(blob=>blob?resolve(blob):reject(new Error('PNG export failed')),'image/png'));
      if(signal.aborted)throw new Error('aborted');
      return callback(image,signal);
    });
  }
          onRecognition(event                  )       {
    this.updateControls();
    if(event.type==='started'){
      this.status.textContent='Recognizing…';this.dispatchEvent(new CustomEvent('recognition-started',{bubbles:true,composed:true}));
    }else if(event.type==='result'){
      this.status.textContent=event.result.status==='recognized'?`Predicted symbol: ${event.result.character}`:'Not recognized. Try drawing again.';
      this.dispatchEvent(new CustomEvent('recognition-result',{detail:event.result,bubbles:true,composed:true}));
    }else if(event.type==='error'){
      this.status.textContent=event.message;this.dispatchEvent(new CustomEvent('recognition-error',{detail:{message:event.message},bubbles:true,composed:true}));
    }else this.status.textContent='';
  }
}
if(!customElements.get('tamil-symbol-input'))customElements.define('tamil-symbol-input',TamilSymbolInput);
