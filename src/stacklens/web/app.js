'use strict';
const $ = id => document.getElementById(id);
let registering=false, sessionId=null, signedIn=false, pollTimer=null, busy=false;
async function api(path, options={}) {
  const headers={'X-StackLens-Request':'1',...(options.headers||{})};
  if(options.body && !(options.body instanceof FormData)) headers['Content-Type']='application/json';
  const response=await fetch('/api/v1'+path,{...options,headers,credentials:'same-origin'});
  const data=await response.json().catch(()=>({detail:'The server returned an unexpected response.'}));
  if(!response.ok){
    if(response.status===401 && !['/login','/register'].includes(path)) showLogin();
    throw new Error(typeof data.detail==='string'?data.detail:'Please check the entered values.');
  }
  return data;
}
function notice(message){$('notification').textContent=message;$('notification').hidden=!message;}
function showLogin(){signedIn=false;clearTimeout(pollTimer);if(!$('auth-dialog').open)$('auth-dialog').showModal();}
function signed(user){signedIn=true;$('username').textContent=user.username;$('avatar').textContent=user.username[0].toUpperCase();$('auth-dialog').close();refreshDocuments();}
$('auth-dialog').addEventListener('cancel',event=>event.preventDefault());
$('auth-toggle').onclick=()=>{registering=!registering;$('auth-title').textContent=registering?'Create your workspace':'Welcome to StackLens';$('auth-submit').textContent=registering?'Create account':'Sign in';$('auth-toggle').textContent=registering?'Already have an account? Sign in':'New here? Create an account';$('auth-password').autocomplete=registering?'new-password':'current-password';$('auth-error').textContent='';};
$('auth-form').onsubmit=async event=>{event.preventDefault();$('auth-submit').disabled=true;$('auth-error').textContent='';try{signed(await api(registering?'/register':'/login',{method:'POST',body:JSON.stringify({username:$('auth-username').value,password:$('auth-password').value})}));$('auth-password').value='';}catch(error){$('auth-error').textContent=error.message;}finally{$('auth-submit').disabled=false;}};
$('logout').onclick=async()=>{try{await api('/logout',{method:'POST'});resetChat();$('documents').replaceChildren();showLogin();}catch(error){notice(error.message);}};
function resetChat(){sessionId=null;for(const el of document.querySelectorAll('.message'))el.remove();$('welcome').hidden=false;notice('');}
$('new-chat').onclick=()=>{if(!busy)resetChat();};
async function refreshDocuments(){
 clearTimeout(pollTimer);if(!signedIn)return;
 try{const data=await api('/documents');if(!signedIn)return;$('documents').replaceChildren();$('doc-count').textContent=data.documents.length;
  if(!data.documents.length){const p=document.createElement('p');p.className='empty-library';p.textContent='Your documents will appear here.';$('documents').append(p);}
  for(const doc of data.documents){const card=document.createElement('div');card.className='document';const title=document.createElement('strong');title.textContent=doc.filename;const status=document.createElement('small');status.textContent=doc.status==='completed'?`${doc.chunks} passages · Ready`:doc.status==='failed'?doc.error:doc.status==='processing'?'Processing document…':'Queued for processing';status.className=doc.status;card.append(title,status);if(doc.status==='failed'){const retry=document.createElement('button');retry.textContent='Retry processing';retry.onclick=async()=>{retry.disabled=true;try{await api(`/documents/${doc.id}/retry`,{method:'POST'});refreshDocuments();}catch(error){notice(error.message);retry.disabled=false;}};card.append(retry);}$('documents').append(card);}
 }catch(error){notice(error.message);}finally{if(signedIn)pollTimer=setTimeout(refreshDocuments,4000);}
}
$('upload').onchange=async()=>{const file=$('upload').files[0];if(!file)return;if(file.size>5000000){notice('Choose a file smaller than 5 MB.');return;}const data=new FormData();data.append('file',file);$('upload').disabled=true;notice('Uploading your document…');try{await api('/documents',{method:'POST',body:data});notice('Document received. Processing status appears in your library.');refreshDocuments();}catch(error){notice(error.message);}finally{$('upload').disabled=false;$('upload').value='';}};
function message(role,text){$('welcome').hidden=true;const box=document.createElement('article');box.className='message '+role;const label=document.createElement('div');label.className='role';label.textContent=role==='user'?'You':'StackLens';const content=document.createElement('div');content.className='content';content.textContent=text;box.append(label,content);$('conversation').append(box);box.scrollIntoView({block:'end'});return box;}
for(const button of document.querySelectorAll('[data-question]'))button.onclick=()=>{$('question').value=button.dataset.question;$('question').focus();};
$('question-form').onsubmit=async event=>{event.preventDefault();if(busy)return;const question=$('question').value.trim();if(!question)return;if(!signedIn){showLogin();return;}busy=true;$('send').disabled=true;$('new-chat').disabled=true;notice('');message('user',question);$('question').value='';const answer=message('assistant','Searching your documents and checking the evidence…');try{const result=await api('/chat',{method:'POST',body:JSON.stringify({question,session_id:sessionId})});sessionId=result.session_id;answer.querySelector('.content').textContent=result.answer;const sources=document.createElement('div');sources.className='sources';for(const name of result.sources){const tag=document.createElement('span');tag.className='source';tag.textContent=name;sources.append(tag);}answer.append(sources);if(result.metrics){const m=document.createElement('div');m.className='metrics';m.textContent=`${result.metrics.latency_seconds.toFixed(1)}s · ${result.metrics.total_tokens} tokens`+(result.metrics.estimated_cost_usd!==null?` · estimated $${result.metrics.estimated_cost_usd.toFixed(5)}`:'');answer.append(m);}}catch(error){answer.querySelector('.content').textContent=error.message;}finally{busy=false;$('send').disabled=false;$('new-chat').disabled=false;answer.scrollIntoView({block:'end'});}};
api('/me').then(signed).catch(()=>showLogin());
