export const TAMIL_SYMBOLS                    = Object.freeze(['அ','ஆ','இ','ஈ','உ','ஊ','எ','ஏ','ஐ','ஒ','ஓ','ஔ','ஃ']);
                                 
                     
                        
                        
                                                                                            
                                                                                         
                                                                                                                                                

export function validateResult(value         )                    {
  if (!value || typeof value !== 'object' || Array.isArray(value)) throw new TypeError('Invalid recognition response');
  const v = value                          ;
  if (typeof v.request_id !== 'string' || v.request_id.length > 200 || !v.request_id ||
      typeof v.model_version !== 'string' || v.model_version.length > 200 || !v.model_version ||
      typeof v.processing_ms !== 'number' || !Number.isFinite(v.processing_ms) || v.processing_ms < 0 ||
      !((v.status === 'recognized' && typeof v.character === 'string' && TAMIL_SYMBOLS.includes(v.character)) ||
        (v.status === 'unrecognized' && v.character === null))) throw new TypeError('Invalid recognition response');
  return {request_id:v.request_id, status:v.status, character:v.character, model_version:v.model_version, processing_ms:v.processing_ms}                     ;
}

export class RecognitionController {
  pending = false;
          sequence = 0;
          aborter                         = null;
          notify                                   ;
  constructor(notify                                   ) {this.notify = notify;}
  cancel()       {
    this.sequence += 1;
    this.aborter?.abort();
    this.aborter = null;
    this.pending = false;
    this.notify({type:'cancelled'});
  }
  async submit(image      , recognize           )                   {
    if (this.pending) return false;
    const sequence = ++this.sequence;
    const aborter = new AbortController();
    this.aborter = aborter;
    this.pending = true;
    this.notify({type:'started'});
    const timeout = setTimeout(()=>aborter.abort(),10000);
    try {
      const aborted = new Promise       ((_resolve,reject)=>aborter.signal.addEventListener('abort',()=>reject(new Error('aborted')),{once:true}));
      const result = validateResult(await Promise.race([recognize(image,aborter.signal),aborted]));
      if (sequence !== this.sequence) return false;
      this.pending = false;
      this.notify({type:'result',result});
      return true;
    } catch {
      if (sequence !== this.sequence) return false;
      this.pending = false;
      this.notify({type:'error',message:aborter.signal.aborted?'Recognition timed out. Please try again.':'Recognition failed. Please try again.'});
      return false;
    } finally {
      clearTimeout(timeout);
      if (sequence === this.sequence) this.aborter = null;
    }
  }
}
