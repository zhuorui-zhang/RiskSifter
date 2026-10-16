const $ = id => document.getElementById(id);
let videos=[], events=[], allEvents=[], selectedVideo='', selectedEvent=null, lastJobState='', status={}, dirty=false;
const labels={pending:'待审核',approved:'已确认',rejected:'已拒绝'};
const esc=s=>String(s??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const time=s=>`${String(Math.floor(s/60)).padStart(2,'0')}:${(s%60).toFixed(1).padStart(4,'0')}`;
async function api(path,options={}){const r=await fetch('/api'+path,options);if(!r.ok){let e=await r.json().catch(()=>({detail:r.statusText}));throw Error(typeof e.detail==='string'?e.detail:JSON.stringify(e.detail));}return r.json();}
const post=(path,body)=>api(path,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});
function toast(message,error=false){$('toast').textContent=message;$('toast').className='toast'+(error?' error':'');clearTimeout(toast.timer);toast.timer=setTimeout(()=>$('toast').classList.add('hidden'),error?12000:4500);}
function guard(fn){return async(...args)=>{try{await fn(...args)}catch(e){toast(e.message,true)}}}
async function refresh(){
 [videos,allEvents]=await Promise.all([api('/videos'),api('/events?latest_only='+($('runScope').value!=='all'))]);
 $('statVideos').textContent=videos.length;$('statEvents').textContent=allEvents.length;$('eventCount').textContent=allEvents.length;
 $('statApproved').textContent=allEvents.filter(e=>e.review_status==='approved').length;$('statPending').textContent=allEvents.filter(e=>e.review_status==='pending').length;
 if(!selectedVideo&&videos.length===1)selectedVideo=videos[0].id;
 renderVideos();await filter();
}
function renderVideos(){
 const mobileSelect=$('videoFilter');
 if(mobileSelect){mobileSelect.innerHTML='<option value="">全部视频</option>'+videos.map(v=>`<option value="${v.id}">${esc(v.name)}</option>`).join('');mobileSelect.value=selectedVideo;}
 $('videoList').innerHTML=videos.map(v=>`<button class="video-item ${v.id===selectedVideo?'selected':''}" data-video="${v.id}" title="${esc(v.name)}">▷ ${esc(v.name)}<small>${v.status==='ready'?time(v.duration)+' · '+v.width+'×'+v.height:esc(v.status+' '+(v.error||''))}</small></button>`).join('');
 $('videoList').querySelectorAll('button').forEach(b=>b.onclick=guard(async()=>{selectedVideo=b.dataset.video;renderVideos();await filter();}));
}
async function filter(){
 const params=new URLSearchParams({q:$('search').value,video_id:selectedVideo,review_status:$('statusFilter').value,latest_only:$('runScope').value!=='all'});
 events=await api('/events?'+params);renderEvents();
}
function renderEvents(){
 $('visibleCount').textContent=events.length+' 个结果';$('emptyState').classList.toggle('hidden',events.length>0);
 if(!events.length){$('emptyState').querySelector('h3').textContent=videos.length?'尚无匹配候选':'从一段行车视频开始';$('emptyState').querySelector('p').textContent=videos.length?'选择视频并开始挖掘，或尝试其他搜索词与筛选条件。':'导入本地视频，选择视频后点击「挖掘」。候选与证据会保存在这里。';$('emptyImport').classList.toggle('hidden',videos.length>0);}
 $('eventList').innerHTML=events.map(e=>`<div tabindex="0" role="button" aria-label="查看 ${time(e.start)} 事件" class="event-card ${selectedEvent?.id===e.id?'selected':''}" data-id="${e.id}"><div class="card-image"><img loading="lazy" src="/api/events/${e.id}/evidence/2" alt="候选证据帧"><span class="badge ${e.review_status}">${labels[e.review_status]}</span><span class="card-time">${time(e.start)} → ${time(e.end)}</span></div><div class="card-body"><small>${esc(videos.find(v=>v.id===e.video_id)?.name||e.video_id)}</small><p>${esc(e.description)}</p><div class="card-tags">${e.tags.slice(0,3).map(t=>`<span>${esc(t)}</span>`).join('')}</div></div></div>`).join('');
 $('eventList').querySelectorAll('.event-card').forEach(c=>{c.onclick=guard(()=>selectEvent(c.dataset.id));c.onkeydown=e=>{if(e.key==='Enter')c.click()}});
 if(selectedEvent&&!events.some(e=>e.id===selectedEvent.id)){selectedEvent=null;dirty=false;$('detail').classList.add('hidden');$('detailEmpty').classList.remove('hidden');}
}
async function selectEvent(id){
 if(dirty&&!confirm('当前修改尚未保存，是否放弃修改？'))return;
 selectedEvent=await api('/events/'+id);dirty=false;renderEvents();renderDetail();
}
function renderDetail(){
 const e=selectedEvent,v=videos.find(v=>v.id===e.video_id);if(!v)return;
 $('detail').classList.remove('hidden');$('detailEmpty').classList.add('hidden');
 $('detailTitle').textContent=v.name;$('reviewBadge').textContent=labels[e.review_status];$('reviewBadge').className='badge '+e.review_status;
 const player=$('player');player.src='/api/videos/'+v.id+'/media';player.onloadedmetadata=()=>{player.currentTime=e.start};
 player.ontimeupdate=()=>{if(!player.paused&&player.currentTime>=Number($('end').value)){player.pause();}};
 $('timeRange').textContent=time(e.start)+' — '+time(e.end)+' / '+time(v.duration);$('clipDownload').href='/api/events/'+e.id+'/clip';
 $('totalDuration').textContent=time(v.duration);
 const related=allEvents.filter(x=>x.video_id===v.id);
 $('timeline').innerHTML=related.map(x=>`<button title="${time(x.start)} — ${time(x.end)}" aria-label="跳转到 ${time(x.start)}" class="${x.id===e.id?'current':''}" data-id="${x.id}" style="left:${100*x.start/v.duration}%;width:${100*(x.end-x.start)/v.duration}%"></button>`).join('');
 $('timeline').querySelectorAll('button').forEach(b=>b.onclick=guard(()=>selectEvent(b.dataset.id)));
 $('evidenceLabel').textContent=e.annotation.model.backend==='none'?'已提取，未被 VLM 查看':'VLM 实际查看的 '+e.evidence.length+' 帧';
 $('evidence').innerHTML=e.evidence.map((item,i)=>`<button title="跳转证据时间" data-time="${item.timestamp}" type="button"><img src="/api/events/${e.id}/evidence/${i}" alt="${item.timestamp.toFixed(2)} 秒证据"><span>${time(item.timestamp)}</span></button>`).join('');
 $('evidence').querySelectorAll('button').forEach(b=>b.onclick=()=>{player.pause();player.currentTime=Number(b.dataset.time)});
 $('uncertainty').textContent=[...e.annotation.uncertainties,...(e.annotation_error?[e.annotation_error]:[]),...(e.boundary_changed?['边界已修改；模型证据仍对应原始候选范围。']:[])].join(' · ');
 for(const key of ['start','end','description','query'])$(key).value=e[key];$('tags').value=e.tags.join(', ');$('reviewNote').value=e.review_note;
 $('provenance').textContent=JSON.stringify({event_id:e.id,source_sha256:v.sha256,source:v.provenance,proposal:e.proposal,annotation:e.annotation,evidence:e.evidence.map(({file,...rest})=>rest),history:e.history},null,2);
}
async function review(review_status){
 if(!selectedEvent)return;
 const body={revision:selectedEvent.revision,start:Number($('start').value),end:Number($('end').value),description:$('description').value,query:$('query').value,tags:$('tags').value.split(',').map(t=>t.trim()).filter(Boolean),review_note:$('reviewNote').value,review_status};
 const saved=await api('/events/'+selectedEvent.id,{method:'PATCH',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});
 dirty=false;selectedEvent=saved;await refresh();if(selectedEvent)renderDetail();toast('已保存审核与修改历史');
}
async function pollJobs(){
 const jobs=await api('/jobs');const visible=jobs.slice(0,4);$('jobsPanel').classList.toggle('hidden',!visible.length);
 $('jobsPanel').innerHTML=visible.map(j=>`<div class="job ${j.status}"><b>${({running:'◌',queued:'◷',completed:'✓',failed:'!',partial:'!',interrupted:'!'})[j.status]||'·'}</b><span>${esc(j.message)}</span>${j.kind==='export'&&j.result?.id?`<a href="/api/exports/${j.result.id}">↓ 下载 ZIP (${j.result.event_count} 个片段)</a>`:`<small>${esc(j.kind)}</small>`}</div>`).join('');
 const signature=jobs.map(j=>j.id+':'+j.status).join('|');if(signature!==lastJobState){lastJobState=signature;await refresh();}
}
const showImport=()=>$('importDialog').showModal();['openImport','importButton','emptyImport'].forEach(id=>$(id).onclick=showImport);
$('pathImport').onclick=guard(async()=>{const path=$('localPath').value.trim();if(!path)throw Error('请输入本机视频或文件夹路径');await post('/import',{path});$('importDialog').close();toast('导入任务已加入队列');await pollJobs()});
async function uploadFiles(files){$('importDialog').close();for(const file of files){toast('正在上传到本机：'+file.name);const data=new FormData();data.append('file',file);await api('/upload',{method:'POST',body:data});}toast('导入任务已加入队列');await pollJobs();}
$('fileInput').onchange=guard(e=>uploadFiles(e.target.files));
$('dropzone').ondragover=e=>{e.preventDefault();$('dropzone').classList.add('drag')};$('dropzone').ondragleave=()=>$('dropzone').classList.remove('drag');$('dropzone').ondrop=guard(async e=>{e.preventDefault();$('dropzone').classList.remove('drag');await uploadFiles(e.dataTransfer.files)});
$('mineButton').onclick=guard(()=>{if(!selectedVideo)throw Error('请先在左侧选择一个已导入的视频');const v=videos.find(v=>v.id===selectedVideo);if(v.status!=='ready')throw Error('视频尚未完成导入');$('mineVideoName').textContent=v.name;$('modelNote').textContent=status.model_available?'已找到本地模型。描述质量需要人工核对。':'尚未找到模型权重。轻量模式可立即使用；模型安装见 README。';$('annotator').value=status.model_available?'smolvlm':'none';$('mineDialog').showModal()});
$('startMine').onclick=guard(async()=>{await post('/mine',{video_id:selectedVideo,annotator:$('annotator').value,max_candidates:Number($('budget').value),window_seconds:Number($('window').value)});$('mineDialog').close();toast('挖掘任务已开始');await pollJobs()});
$('reviewForm').onsubmit=guard(async e=>{e.preventDefault();await review('approved')});$('rejectButton').onclick=guard(()=>review('rejected'));$('saveButton').onclick=guard(()=>review('pending'));
$('reviewForm').oninput=()=>{dirty=true};$('setStart').onclick=()=>{$('start').value=$('player').currentTime.toFixed(3);dirty=true};$('setEnd').onclick=()=>{$('end').value=$('player').currentTime.toFixed(3);dirty=true};
$('exportButton').onclick=()=>$('exportDialog').showModal();$('startExport').onclick=guard(async()=>{await post('/exports',{include_pending:$('includePending').checked,event_ids:events.map(e=>e.id)});$('exportDialog').close();toast('正在生成数据集，完成后会出现下载链接');await pollJobs()});
$('statusFilter').onchange=guard(filter);let debounce;$('search').oninput=()=>{clearTimeout(debounce);debounce=setTimeout(guard(filter),250)};
$('runScope').onchange=guard(refresh);
$('allVideos').onclick=guard(async()=>{selectedVideo='';renderVideos();await filter()});document.onkeydown=e=>{if((e.metaKey||e.ctrlKey)&&e.key==='k'){e.preventDefault();$('search').focus()}};
if($('videoFilter'))$('videoFilter').onchange=guard(async()=>{selectedVideo=$('videoFilter').value;renderVideos();await filter()});
window.onbeforeunload=e=>{if(dirty){e.preventDefault();e.returnValue=''}};
guard(async()=>{status=await api('/status');$('hardware').textContent=status.memory_gb+' GB 内存 · '+(status.model_available?'本地 VLM 就绪':'轻量模式可用');await refresh();await pollJobs();setInterval(guard(pollJobs),2500)})();
