"use strict";
const $=s=>document.querySelector(s), $$=s=>[...document.querySelectorAll(s)];
const clamp=(v,a,b)=>Math.min(b,Math.max(a,v)), sleep=ms=>new Promise(r=>setTimeout(r,ms));
const fmt=t=>{t=Math.max(0,t|0);return Math.floor(t/60)+':'+String(t%60).padStart(2,'0')};
const esc=s=>String(s).replace(/[&<>"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));
const UI_FONT='"Be Vietnam Pro",system-ui,-apple-system,sans-serif';
const BEH={
  touching:{c:'#A36800',t:'Chạm hiện vật'}, holding:{c:'#C95214',t:'Cầm / nhấc'}, stealing:{c:'#C2272D',t:'Nghi lấy cắp'},
  candidate:{c:'#5542C4',t:'Cần xác nhận'}, normal:{c:'#636C74',t:'Bình thường'}
};
const evColor=e=>e.decision==='ANOMALY'?(BEH[e.behavior]||BEH.stealing).c:e.decision==='CANDIDATE'?BEH.candidate.c:BEH.normal.c;
const evLabel=e=>e.decision==='ANOMALY'?(BEH[e.behavior]||BEH.stealing).t:e.decision==='CANDIDATE'?BEH.candidate.t:BEH.normal.t;

/* ---------- Dữ liệu mẫu (cùng schema backend sẽ trả) ---------- */
function makeMock(D=60){
  const k=D/60, S=v=>v*k;
  const arts=[{id:1,name:'Bình gốm Bát Tràng',fx:.2,fy:.35},{id:2,name:'Tượng Chăm',fx:.78,fy:.3},
              {id:3,name:'Tranh sơn mài',fx:.25,fy:.78},{id:4,name:'Trống đồng',fx:.75,fy:.75}];
  const P={
    1:[[0,.5,.97],[8,.5,.6],[20,.55,.55],[36,.8,.65],[40,.75,.82],[43,.75,.82],[52,.5,.7],[60,.15,.97]],
    2:[[2,.05,.9],[10,.2,.6],[14,.2,.42],[19,.2,.42],[28,.45,.4],[36,.6,.3],[44,.95,.5]],
    3:[[8,.95,.9],[16,.85,.6],[22,.75,.82],[26,.75,.82],[34,.5,.9],[42,.3,.95],[48,.1,.97]],
    4:[[18,.05,.5],[26,.15,.7],[31,.25,.85],[38,.25,.85],[46,.45,.6],[56,.5,.97]],
    5:[[30,.6,.97],[38,.7,.6],[47,.78,.37],[53,.78,.37],[56,.92,.6],[60,.99,.9]],
    6:[[0,.9,.5],[12,.6,.45],[24,.5,.3],[40,.35,.45],[60,.05,.5]],
    7:[[34,.1,.95],[44,.35,.55],[58,.6,.4]]};
  const tracks=Object.entries(P).map(([id,w])=>({id:+id,pts:w.map(p=>[S(p[0]),p[1],p[2]])}));
  const ev=(id,track,art,a,b,behavior,decision,conf,reason)=>({id,track,art,start:S(a),end:S(b),behavior,decision,conf,reason,review:null});
  const events=[
    ev(1,2,1,14,19,'touching','ANOMALY',.91,'Bàn tay phải đặt lên thân bình gốm khoảng 5 giây; khu vực không có biển cho phép chạm.'),
    ev(2,4,3,31,38,'holding','ANOMALY',.87,'Khách nhấc khung tranh khỏi giá treo bằng hai tay trong nhiều giây.'),
    ev(3,5,2,47,53,'stealing','ANOMALY',.93,'Khách đưa vật nhỏ vào túi áo khoác, nhìn quanh trước khi rời đi nhanh.'),
    ev(4,3,4,22,26,'uncertain','CANDIDATE',.52,'Tay ở gần trống đồng nhưng bị che một phần, chưa đủ chắc chắn để kết luận.'),
    ev(5,1,4,40,43,'normal','NORMAL',.88,'Chỉ đứng xem và chỉ tay về phía hiện vật, không có tiếp xúc.')];
  return {meta:{sequence_id:'demo_01',camera_id:'cam_01',duration:D,fps:25,floor:{w:8,h:6,unit:'m'}},arts,tracks,events};
}

/* ---------- Chuyển result.json của pipeline sang dữ liệu web ---------- */
function fromResult(res,imgFor){
  const v=res.video,fps=v.fps||25,nw=Math.min((((res.config||{}).thresholds||{}).CFG||{}).width||960,v.width);
  const arts=(res.artifacts||[]).map(a=>({id:a.id,name:a.label||('Hiện vật '+a.id),box:a.box,pedestal:!!a.pedestal}));
  const al=Object.fromEntries((res.alerts||[]).map(a=>[a.event_id,a]));
  const why={NORMAL:'VLM đánh giá hành vi bình thường.',CANDIDATE:'VLM chưa đủ chắc chắn để kết luận.',ANOMALY:'VLM kết luận có hành vi bất thường.'};
  const events=(res.events||[]).map(e=>{const a=al[e.event_id];
    return{id:e.event_id,track:e.track_id,art:e.artifact_id,start:e.start_time_s,end:e.end_time_s,
      behavior:e.decision==='CANDIDATE'?'uncertain':e.behavior,decision:e.decision,conf:e.vlm_confidence==null?(a?a.confidence:null):e.vlm_confidence,
      reason:(a&&a.evidence)||e.vlm_evidence||why[e.decision]||'',review:null,img:imgFor?imgFor(e):null}});
  return{real:true,raw:res,meta:{sequence_id:v.key,camera_id:v.name,duration:v.n_frames/fps,fps,width:v.width,height:v.height,nw},arts,tracks:[],events};
}

/* ---------- Tính toán dẫn xuất ---------- */
let DATA=null, D=60, DER=null;
function posAt(tr,t){const p=tr.pts;if(t<p[0][0]||t>p[p.length-1][0])return null;
  let i=1;while(i<p.length-1&&p[i][0]<t)i++;const a=p[i-1],b=p[i],u=b[0]===a[0]?0:(t-a[0])/(b[0]-a[0]);
  return{fx:a[1]+(b[1]-a[1])*u+Math.sin(t*1.7+tr.id)*.004,fy:a[2]+(b[2]-a[2])*u+Math.cos(t*1.3+tr.id)*.004}}
function activeAt(t){const o=[];for(const tr of DATA.tracks){const p=posAt(tr,t);if(p)o.push({id:tr.id,...p})}return o}
function derive(){
  const W=48,H=36,heat=new Float32Array(W*H),dwell={},journey={},occ=[];
  DATA.arts.forEach(a=>dwell[a.id]={total:0,vis:new Set()});
  for(const tr of DATA.tracks){journey[tr.id]=[];const seen={};
    for(let t=tr.pts[0][0];t<=tr.pts[tr.pts.length-1][0];t+=.5*D/60){const p=posAt(tr,t);if(!p)continue;
      const gx=p.fx*(W-1),gy=p.fy*(H-1);
      for(let y=Math.max(0,gy-3|0);y<Math.min(H,gy+4);y++)for(let x=Math.max(0,gx-3|0);x<Math.min(W,gx+4);x++)heat[y*W+x]+=Math.exp(-((x-gx)**2+(y-gy)**2)/3);
      for(const a of DATA.arts)if(Math.hypot(p.fx-a.fx,p.fy-(a.fy+.07))<.1){dwell[a.id].total+=.5*D/60;dwell[a.id].vis.add(tr.id);
        (seen[a.id]=seen[a.id]||{art:a.id,t0:t,dur:0}).dur+=.5*D/60}}
    journey[tr.id]=Object.values(seen).filter(s=>s.dur>=2*D/60).sort((a,b)=>a.t0-b.t0)}
  for(let t=0;t<D;t+=D/60)occ.push(activeAt(t).length);
  let mx=1e-6;heat.forEach(v=>mx=Math.max(mx,v));
  DER={W,H,heat,mx,dwell,journey,occ};
}

/* ---------- Vẽ chung ---------- */
function fit(c){const r=c.getBoundingClientRect(),d=window.devicePixelRatio||1,w=Math.round(r.width*d),h=Math.round(r.height*d);
  if(c.width!==w||c.height!==h){c.width=w;c.height=h}const x=c.getContext('2d');x.setTransform(d,0,0,d,0,0);return{x,w:r.width,h:r.height}}
const proj=(fx,fy)=>{const s=.45+.55*fy;return{x:.5+(fx-.5)*s,y:.24+.7*fy,s}};
function rr(x,X,Y,W,H,R){x.beginPath();x.roundRect(X,Y,W,H,R)}
function activeEvents(t){return DATA.events.filter(e=>t>=e.start&&t<=e.end)}

function drawReal(t){
  const {x,w,h}=fit($('#scene'));if(w<30||h<30)return;x.clearRect(0,0,w,h);
  const M=DATA.meta,hasVid=!!S.videoURL,vw=(hasVid&&vid.videoWidth)||M.width,vh=(hasVid&&vid.videoHeight)||M.height;
  const sc=Math.min(w/vw,h/vh),ox=(w-vw*sc)/2,oy=(h-vh*sc)/2,k=vw*sc/M.nw;
  if(!hasVid){x.fillStyle='#141A20';x.fillRect(ox,oy,vw*sc,vh*sc);x.fillStyle='#A9B4BA';x.font='500 13px '+UI_FONT;x.textAlign='center';
    x.fillText('Chưa chọn video — chỉ hiển thị cảnh báo và vị trí hiện vật',w/2,h/2);x.textAlign='left'}
  const aev=activeEvents(t);
  for(const a of DATA.arts){const e=aev.find(e=>e.art===a.id&&e.decision!=='NORMAL'),n=aev.find(e=>e.art===a.id);
    if((!S.layers.art&&!e)||(a.pedestal&&!e))continue;
    const [x1,y1,x2,y2]=a.box,X=ox+x1*k,Y=oy+y1*k,col=e?evColor(e):n?'#636C74':'#1B64B8';
    x.strokeStyle=col;x.lineWidth=e?3:2;x.strokeRect(X,Y,(x2-x1)*k,(y2-y1)*k);
    if(S.layers.labels||e){const txt=e?`${a.name} · ${evLabel(e)}`:a.name;x.font='600 11px '+UI_FONT;const tw=x.measureText(txt).width+10;
      x.fillStyle=col;x.fillRect(X,Y-18,tw,18);x.fillStyle='#fff';x.fillText(txt,X+5,Y-5)}}
}
function drawScene(t){
  if(DATA.real)return drawReal(t);
  const {x,w,h}=fit($('#scene')),hasVid=!!S.videoURL;
  if(w<30||h<30)return;
  x.clearRect(0,0,w,h);
  if(!hasVid){
    x.fillStyle='#111920';x.fillRect(0,0,w,h);
    const g=x.createLinearGradient(0,0,0,h*.26);g.addColorStop(0,'#1f2b34');g.addColorStop(1,'#2a3843');x.fillStyle=g;x.fillRect(0,0,w,h*.24);
    const f=x.createLinearGradient(0,h*.24,0,h);f.addColorStop(0,'#18232b');f.addColorStop(1,'#111920');x.fillStyle=f;x.fillRect(0,h*.24,w,h);
    x.strokeStyle='rgba(255,255,255,.05)';x.lineWidth=1;
    for(let i=0;i<=10;i++){const a=proj(i/10,0),b=proj(i/10,1);x.beginPath();x.moveTo(a.x*w,a.y*h);x.lineTo(b.x*w,b.y*h);x.stroke()}
    for(let j=0;j<=8;j++){const a=proj(0,j/8),b=proj(1,j/8);x.beginPath();x.moveTo(a.x*w,a.y*h);x.lineTo(b.x*w,b.y*h);x.stroke()}
    for(const a of DATA.arts){const p=proj(a.fx,a.fy),pw=w*.075*p.s,ph=h*.11*p.s;
      x.fillStyle='#33434f';x.fillRect(p.x*w-pw/2,p.y*h-ph,pw,ph);x.fillStyle='#cbd5ee';
      x.beginPath();x.ellipse(p.x*w,p.y*h-ph-pw*.28,pw*.3,pw*.34,0,0,7);x.fill()}
  }
  const act=activeAt(t),aev=activeEvents(t),L=S.layers;
  // vẽ silhouette khi chưa có video thật
  const boxOf=p=>{const q=proj(p.fx,p.fy),bh=h*.34*q.s,bw=bh*.42;return{x:q.x*w-bw/2,y:q.y*h-bh,w:bw,h:bh,fx:q.x*w,fy:q.y*h}};
  if(!hasVid)for(const p of act){const b=boxOf(p);x.fillStyle='rgba(165,180,210,.28)';rr(x,b.x+b.w*.12,b.y+b.h*.18,b.w*.76,b.h*.8,b.w*.3);x.fill();
    x.beginPath();x.arc(b.x+b.w/2,b.y+b.h*.09,b.w*.2,0,7);x.fill()}
  if(L.art)for(const a of DATA.arts){const p=proj(a.fx,a.fy),pw=w*.085*p.s,ph=h*.15*p.s;x.strokeStyle='#1B64B8';x.lineWidth=2;
    x.strokeRect(p.x*w-pw/2,p.y*h-ph-pw*.3,pw,ph+pw*.3);x.fillStyle='#5B9BEA';x.font='600 11px '+UI_FONT;x.fillText('A'+a.id+' '+a.name,p.x*w-pw/2,p.y*h-ph-pw*.3-5)}
  for(const p of act){const b=boxOf(p),e=aev.find(e=>e.track===p.id&&e.decision!=='NORMAL');
    const col=e?evColor(e):'#2EB872';
    if(L.skel){skeleton(x,b,p,e,t,col)}
    if(L.boxes){x.strokeStyle=col;x.lineWidth=e?3:2;x.strokeRect(b.x,b.y,b.w,b.h);
      if(L.labels){const txt='Visitor #'+p.id+(e?' · '+(e.decision==='CANDIDATE'?'?':BEH[e.behavior].t)+' '+(e.conf*100|0)+'%':'');
        x.font='600 11px '+UI_FONT;const tw=x.measureText(txt).width+10;x.fillStyle=col;x.fillRect(b.x,b.y-18,tw,18);x.fillStyle='#fff';x.fillText(txt,b.x+5,b.y-5)}}}
}
function skeleton(x,b,p,e,t,col){
  const sw=Math.sin(t*6+p.id)*.06,K={h:[.5,.09],n:[.5,.2],ls:[.28,.23],rs:[.72,.23],le:[.2,.42],re:[.8,.42],lw:[.22,.58],rw:[.78,.58],lh:[.38,.55],rh:[.62,.55],lk:[.4+sw,.77],rk:[.6-sw,.77],la:[.4-sw,.97],ra:[.6+sw,.97]};
  const P={};for(const k in K)P[k]=[b.x+K[k][0]*b.w,b.y+K[k][1]*b.h];
  if(e){const a=DATA.arts.find(a=>a.id===e.art),q=proj(a.fx,a.fy),W=x.canvas.width/(window.devicePixelRatio||1),H=x.canvas.height/(window.devicePixelRatio||1);
    const u=Math.min(1,(t-e.start)/1.2),tx=q.x*W,ty=q.y*H-H*.1*q.s;
    P.rw=[P.rw[0]+(tx-P.rw[0])*u,P.rw[1]+(ty-P.rw[1])*u];P.re=[(P.rs[0]+P.rw[0])/2+8,(P.rs[1]+P.rw[1])/2]}
  x.strokeStyle=col;x.lineWidth=2;x.lineCap='round';
  [['n','ls'],['n','rs'],['ls','le'],['le','lw'],['rs','re'],['re','rw'],['ls','lh'],['rs','rh'],['lh','rh'],['lh','lk'],['lk','la'],['rh','rk'],['rk','ra']].forEach(([a,c])=>{x.beginPath();x.moveTo(...P[a]);x.lineTo(...P[c]);x.stroke()});
  x.fillStyle='#fff';for(const k in P){x.beginPath();x.arc(P[k][0],P[k][1],2.2,0,7);x.fill()}
  if(e){x.fillStyle=col;x.beginPath();x.arc(P.rw[0],P.rw[1],6,0,7);x.fill()}
}

function floorRect(w,h){const pad=14,ar=4/3;let W=Math.max(10,w-pad*2),H=W/ar;if(H>Math.max(10,h-pad*2)){H=Math.max(10,h-pad*2);W=H*ar}return{x:(w-W)/2,y:(h-H)/2,w:W,h:H}}
function drawMap(c,o){
  const {x,w,h}=fit(c);if(w<30||h<30)return;
  const R=floorRect(w,h),X=fx=>R.x+fx*R.w,Y=fy=>R.y+fy*R.h;
  x.clearRect(0,0,w,h);
  x.fillStyle='#FFFFFF';rr(x,R.x,R.y,R.w,R.h,6);x.fill();x.strokeStyle='#39424A';x.lineWidth=1.75;x.stroke();
  x.strokeStyle='rgba(18,21,24,.065)';x.lineWidth=1;for(let i=1;i<8;i++){x.beginPath();x.moveTo(X(i/8),R.y);x.lineTo(X(i/8),R.y+R.h);x.stroke()}
  for(let j=1;j<6;j++){x.beginPath();x.moveTo(R.x,Y(j/6));x.lineTo(R.x+R.w,Y(j/6));x.stroke()}
  x.strokeStyle='#39424A';x.lineWidth=2.5;x.beginPath();x.moveTo(X(.5),R.y);x.lineTo(X(.5),Y(.14));x.moveTo(X(.5),Y(.26));x.lineTo(X(.5),Y(.5));x.stroke(); // vách ngăn + cửa
  if(o.heat){const cw=R.w/DER.W,ch=R.h/DER.H;for(let j=0;j<DER.H;j++)for(let i=0;i<DER.W;i++){const v=DER.heat[j*DER.W+i]/DER.mx;if(v<.04)continue;
    x.fillStyle=`hsla(${30-v*22},92%,${50+v*12}%,${.1+v*.5})`;x.fillRect(R.x+i*cw-cw/2,R.y+j*ch-ch/2,cw+1,ch+1)}}
  // camera + FOV
  const cx=X(.5),cy=R.y+R.h+1;x.fillStyle='rgba(232,93,26,.065)';x.beginPath();x.moveTo(cx,cy);x.lineTo(X(.02),Y(.0));x.lineTo(X(.98),Y(.0));x.closePath();x.fill();
  x.fillStyle='#E85D1A';x.beginPath();x.arc(cx,Math.min(cy,R.y+R.h),5.5,0,7);x.fill();
  x.strokeStyle='#FFFFFF';x.lineWidth=1.5;x.stroke();
  for(const a of DATA.arts){if(o.zones){x.setLineDash([4,4]);x.strokeStyle='rgba(27,100,184,.38)';x.lineWidth=1.2;x.beginPath();x.arc(X(a.fx),Y(a.fy),R.w*.1,0,7);x.stroke();x.setLineDash([])}
    x.fillStyle='#1B64B8';rr(x,X(a.fx)-6,Y(a.fy)-6,12,12,3);x.fill();x.font='600 10.5px '+UI_FONT;x.textAlign='center';const nw=x.measureText(a.name).width+12;
    x.fillStyle='rgba(255,255,255,.94)';rr(x,X(a.fx)-nw/2,Y(a.fy)-29,nw,17,4);x.fill();x.strokeStyle='rgba(18,21,24,.12)';x.lineWidth=1;x.stroke();
    x.fillStyle='#121518';x.fillText(a.name,X(a.fx),Y(a.fy)-17);x.textAlign='left'}
  if(o.path){const tr=DATA.tracks.find(t=>t.id===o.path);x.strokeStyle='#E85D1A';x.lineWidth=3;x.lineJoin='round';x.beginPath();
    tr.pts.forEach((p,i)=>i?x.lineTo(X(p[1]),Y(p[2])):x.moveTo(X(p[1]),Y(p[2])));x.stroke();
    const s=tr.pts[0];x.fillStyle='#157347';x.beginPath();x.arc(X(s[1]),Y(s[2]),6,0,7);x.fill();
    DER.journey[o.path].forEach((s,i)=>{const a=DATA.arts.find(a=>a.id===s.art);x.fillStyle='#E85D1A';x.beginPath();x.arc(X(a.fx),Y(a.fy+.07),11,0,7);x.fill();x.fillStyle='#FFFFFF';x.font='700 11.5px '+UI_FONT;x.textAlign='center';x.fillText(i+1,X(a.fx),Y(a.fy+.07)+4);x.textAlign='left'})}
  if(o.live){const t=o.t,aev=activeEvents(t);
    for(const tr of DATA.tracks){const p=posAt(tr,t);if(!p)continue;
      if(o.trails){x.strokeStyle='rgba(21,115,71,.42)';x.lineWidth=2;x.beginPath();let st=false;
        for(let u=Math.max(tr.pts[0][0],t-6*D/60);u<=t;u+=.4*D/60){const q=posAt(tr,u);if(!q)continue;st?x.lineTo(X(q.fx),Y(q.fy)):(x.moveTo(X(q.fx),Y(q.fy)),st=true)}x.stroke()}
      const e=aev.find(e=>e.track===tr.id&&e.decision!=='NORMAL'),col=e?evColor(e):'#157347';
      if(e){const r=10+(Math.sin(t*8)+1)*5;x.strokeStyle=col;x.lineWidth=1.8;x.globalAlpha=.55;x.beginPath();x.arc(X(p.fx),Y(p.fy),r,0,7);x.stroke();x.globalAlpha=1}
      x.fillStyle=col;x.beginPath();x.arc(X(p.fx),Y(p.fy),5.5,0,7);x.fill();x.strokeStyle='#FFFFFF';x.lineWidth=1.5;x.stroke();x.fillStyle='#121518';x.font='600 10.5px '+UI_FONT;x.fillText('#'+tr.id,X(p.fx)+9,Y(p.fy)+3.5)}}
}

/* ===================== Dashboard ===================== */
const CFG=window.EF_CONFIG||{demo:false};
const S={t:0,playing:false,speed:1,videoURL:null,file:null,filter:'all',alertScope:'current',loaded:false,running:false,cancel:false,jobId:null,
  progress:0,srcName:'',layers:{boxes:true,skel:false,labels:true,art:false},mlayers:{trails:true,zones:true,heat:false,all:false},mapTab:'events',heatMode:'presence'};
const vid=$('#vid');
let LIVE=null;
let GLOBAL_ALERTS=[];
let JOBS_FILTER='all';
let globalSSE=null;
S.mode=(()=>{try{return localStorage.getItem('ef_mode')==='rt'?'rt':'batch'}catch{return 'batch'}})();S.liveSim=false;
const EMPTY={meta:{duration:60,camera_id:'',width:1280,height:720,nw:960},arts:[],tracks:[],events:[]};

/* ---------- Backend ---------- */
const NGROK_H={'ngrok-skip-browser-warning':'1'};
async function errText(r){try{const j=await r.json();return j.detail||JSON.stringify(j)}catch{return r.status+' '+r.statusText}}
let sseErrorCount=0;
let sseFallbackPolling=false;

const API={
  direct:(()=>{try{return localStorage.getItem('ef_direct')==='1'}catch{return false}})(),
  base:(()=>{
    try{
      if(localStorage.getItem('ef_direct')==='1')return '';
      const stored=localStorage.getItem('ef_api');
      if(stored!==null&&stored!=='')return stored;
      if(location.protocol!=='file:')return '';
      return '';
    }catch{return ''}
  })(),
  key:(()=>{
    try{
      return (sessionStorage.getItem('ef_be_key')||localStorage.getItem('ef_be_key')||'').trim();
    }catch{return ''}
  })(),
  rememberKey:(()=>{
    try{return localStorage.getItem('ef_be_remember_key')==='1'}catch{return false}
  })(),
  ok:false,
  authRequired:false,
  authenticated:true,
  authError:false,
  streamTicket:'',
  streamTicketExpiresAt:0,
  _connState:'',
  _pingSeq:0,
  _sessionEpoch:0,
  _gpuSeq:0,
  _gpuOnline:false,
  _gpuUrl:'',
  _sitesNeedRetry:false,

  isReady(){return this.ok&&(!this.authRequired||this.authenticated)},

  async f(p,o={}){
    const headers={...NGROK_H,...(o.headers||{})};
    if(this.key){
      headers['Authorization']='Bearer '+this.key;
    }
    const url=(p.startsWith('http://')||p.startsWith('https://'))?p:(this.base+p);
    const r=await fetch(url,{...o,headers});
    if(r.status===401){
      this._pingSeq++;
      this._sessionEpoch++;
      this.authRequired=true;
      this.authError=true;
      this.authenticated=false;
      this.streamTicket='';
      this.streamTicketExpiresAt=0;
      this._gpuOnline=false;
      this._gpuUrl='';
      this._sitesNeedRetry=false;
      this._connState='auth';
      disconnectGlobalSSE();
      updateChipAndBanner({ok:true,auth_required:true,authenticated:false});
      const bmsg=$('#beMsg');if(bmsg){bmsg.textContent='yêu cầu mã truy cập';bmsg.style.color='var(--steal)'}
      const gmsg=$('#gpuMsg');if(gmsg){gmsg.textContent='chờ xác thực backend';gmsg.style.color='var(--muted)'}
      if(typeof loadSites==='function')loadSites();
      throw new Error('401 Chưa xác thực: Cần nhập Mã truy cập Backend (API Key)');
    }
    if(!r.ok)throw new Error(await errText(r));
    return r;
  },

  async ensureStreamTicket(){
    if(!this.authRequired)return "";
    if(this.streamTicket&&this.streamTicketExpiresAt&&Date.now()<this.streamTicketExpiresAt-60000){
      return this.streamTicket;
    }
    if(!this.isReady())return "";
    try{
      const r=await this.f('/api/auth/stream-token',{method:'POST'});
      const j=await r.json();
      this.authRequired=!!j.auth_required;
      if(!this.authRequired){
        this.streamTicket="";
        return "";
      }
      this.streamTicket=j.ticket||"";
      this.streamTicketExpiresAt=j.expires_at?(j.expires_at*1000):(Date.now()+(j.expires_in||600)*1000);
      return this.streamTicket;
    }catch{
      return this.streamTicket||"";
    }
  },

  async mediaUrl(path){
    if(!path)return '';
    const base=this.base||'';
    let url=(path.startsWith('http://')||path.startsWith('https://')||path.startsWith('blob:')||path.startsWith('data:'))?path:(base+(path.startsWith('/')?'':'/')+path);
    if(url.startsWith('blob:')||url.startsWith('data:'))return url;
    if(!this.authRequired)return url;
    const ticket=await this.ensureStreamTicket();
    if(!ticket)return url;
    const sep=url.includes('?')?'&':'?';
    return url+sep+'st='+encodeURIComponent(ticket);
  },

  async ping(forceRefresh=false){
    const pingSeq=++this._pingSeq;
    const startGpuSeq=this._gpuSeq;
    let nextOk=false,health=null;
    let nextAuthRequired=false,nextAuthenticated=true,nextAuthError=false;
    const headers={...NGROK_H};
    if(this.key)headers['Authorization']='Bearer '+this.key;

    try{
      const target=(this.base||'')+'/api/health';
      const r=await fetch(target,{headers,signal:AbortSignal.timeout(6000)});
      if(r.ok){
        health=await r.json();
        if(health&&typeof health==='object'&&health.ok===true){
          nextOk=true;
          nextAuthRequired=!!health.auth_required;
          nextAuthenticated=health.authenticated!==false;
          nextAuthError=nextAuthRequired&&!nextAuthenticated;
        }
      }
    }catch{}

    if(pingSeq!==this._pingSeq)return;

    this.ok=nextOk;
    if(nextOk){
      this.authRequired=nextAuthRequired;
      this.authenticated=nextAuthenticated;
      this.authError=nextAuthError;
    }else{
      this.authError=false;
    }

    const c=$('#beChip');
    const bmsg=$('#beMsg');
    const nextState=(!this.ok||!health)?'offline':(this.authRequired&&!this.authenticated)?'auth':'ready';
    const stateChanged=nextState!==this._connState;
    this._connState=nextState;

    if(nextState==='offline'){
      if(stateChanged)this._sessionEpoch++;
      this.authError=false;
      this._gpuOnline=false;
      this._gpuUrl='';
      this._sitesNeedRetry=false;
      c.classList.remove('ok','warn','bad');
      c.classList.add('bad');c.querySelector('span').textContent='Mất kết nối Backend';
      c.title='Mở Cài đặt để kết nối backend';
      if(bmsg){
        bmsg.textContent=this.direct?'Backend cùng origin không phản hồi':(this.base?'không kết nối được':'chưa kết nối (nhập URL Backend)');
        bmsg.style.color='var(--muted)';
      }
      const gmsg=$('#gpuMsg');if(gmsg){gmsg.textContent='chưa kết nối backend';gmsg.style.color='var(--muted)'}
      disconnectGlobalSSE();
      updateChipAndBanner(null);
      if((stateChanged||forceRefresh)&&typeof loadSites==='function')loadSites();
      if((stateChanged||forceRefresh)&&typeof loadJobs==='function')loadJobs();
      if((stateChanged||forceRefresh)&&S.alertScope==='global')renderAlerts();
    }else if(nextState==='auth'){
      if(stateChanged)this._sessionEpoch++;
      this.streamTicket='';
      this.streamTicketExpiresAt=0;
      this._gpuOnline=false;
      this._gpuUrl='';
      this._sitesNeedRetry=false;
      c.classList.remove('ok','warn','bad');
      c.classList.add('bad');c.querySelector('span').textContent='Cần xác thực Backend';
      c.title='Mở Cài đặt để nhập Mã truy cập Backend (API Key)';
      if(bmsg){bmsg.textContent='yêu cầu mã truy cập';bmsg.style.color='var(--steal)'}
      const gmsg=$('#gpuMsg');if(gmsg){gmsg.textContent='chờ xác thực backend';gmsg.style.color='var(--muted)'}
      disconnectGlobalSSE();
      updateChipAndBanner(health);
      if((stateChanged||forceRefresh)&&typeof loadSites==='function')loadSites();
      if((stateChanged||forceRefresh)&&typeof loadJobs==='function')loadJobs();
      if((stateChanged||forceRefresh)&&S.alertScope==='global')renderAlerts();
    }else{
      const gpuStale=this._gpuSeq!==startGpuSeq;
      if(bmsg){bmsg.textContent='đã kết nối ✓';bmsg.style.color='var(--ok)'}
      let gpuCameOnline=false;
      if(!gpuStale){
        gpuCameOnline=syncGpuStatus(health.gpu);
        updateChipAndBanner(health);
      }
      connectGlobalSSE();
      if(stateChanged||forceRefresh){
        loadGpuConfig();
        if(typeof loadSites==='function')loadSites();
        if(typeof loadJobs==='function')loadJobs();
        if(S.alertScope==='global')renderAlerts();
      }else if((!gpuStale&&gpuCameOnline)||(this._sitesNeedRetry&&this._gpuOnline)){
        if(typeof loadSites==='function')loadSites();
      }
    }
    pollSystem();
  },
  async system(){return (await this.f('/api/system')).json()},
  async start(fd){return (await (await this.f('/api/jobs',{method:'POST',body:fd})).json()).job_id},
  async job(id){return (await this.f(`/api/jobs/${id}`)).json()},
  async result(id){return (await this.f(`/api/jobs/${id}/result`)).json()},
  async cancel(id){try{await this.f(`/api/jobs/${id}/cancel`,{method:'POST'})}catch{}},
  async blob(id,p){try{return URL.createObjectURL(await (await this.f(`/api/jobs/${id}/${p}`)).blob())}catch{return null}}
};

function updateChipAndBanner(health){
  const c=$('#beChip');
  const banner=$('#connBanner');
  const bText=$('#connBannerText');

  if(!API.ok||!health){
    c.className='chip bad';
    c.querySelector('span').textContent='Mất kết nối Backend';
    c.title='Mở Cài đặt để kết nối backend';
    if(banner){
      banner.hidden=false;
      banner.className='conn-banner bad';
      if(bText)bText.textContent='Mất kết nối Backend hoặc URL Cloudflare Tunnel đã đổi. Hãy kiểm tra terminal chạy backend/launch.py, sao chép URL HTTPS mới và dán vào Cài đặt.';
    }
    return;
  }

  if(API.authRequired&&!API.authenticated){
    c.className='chip bad';
    c.querySelector('span').textContent='Cần xác thực Backend';
    c.title='Mở Cài đặt để nhập Mã truy cập Backend (API Key)';
    if(banner){
      banner.hidden=false;
      banner.className='conn-banner bad';
      if(bText)bText.textContent='Máy chủ Backend yêu cầu xác thực. Vui lòng nhập Mã truy cập Backend (API Key) trong Cài đặt.';
    }
    return;
  }

  const gpu=health.gpu||{};
  const isGpuOnline=!!gpu.online;
  const isGpuReady=(gpu.ready!==undefined)?!!gpu.ready:(isGpuOnline&&(gpu.demo||(gpu.models&&gpu.models.state==='ready')));
  const modelsState=(gpu.models&&gpu.models.state)||'';

  c.classList.remove('ok','warn','bad');

  if(!isGpuOnline){
    c.classList.add('warn');
    c.querySelector('span').textContent='Backend sẵn sàng · GPU ngoại tuyến';
    c.title='Xem lại lịch sử và cảnh báo vẫn hoạt động khi GPU ngoại tuyến. Bấm để cấu hình GPU.';
    if(banner){
      banner.hidden=false;
      banner.className='conn-banner warn';
      if(bText)bText.textContent='GPU Worker đang ngoại tuyến hoặc đã đổi URL. Lịch sử phân tích & cảnh báo vẫn hoạt động bình thường. Để phân tích video mới, kiểm tra GPU worker hoặc cập nhật URL trong Cài đặt.';
    }
  }else if(!isGpuReady){
    c.classList.add('warn');
    if(modelsState==='loading'){
      c.querySelector('span').textContent='Backend sẵn sàng · GPU đang nạp model';
    }else{
      c.querySelector('span').textContent='Backend sẵn sàng · GPU lỗi model';
    }
    c.title=`GPU Worker: ${(gpu.models&&gpu.models.msg)||'đang tải mô hình'}`;
    if(banner){
      banner.hidden=false;
      banner.className='conn-banner warn';
      if(bText)bText.textContent=`GPU Worker đang tải mô hình (${(gpu.models&&gpu.models.msg)||'vui lòng chờ…'}).`;
    }
  }else{
    c.classList.add('ok');
    let txt='Backend + GPU sẵn sàng';
    if(gpu.busy)txt='Backend + GPU sẵn sàng (GPU đang bận)';
    else if(gpu.demo)txt='Backend + GPU sẵn sàng (Demo)';
    c.querySelector('span').textContent=txt;
    c.title=`GPU Worker trực tuyến tại ${gpu.url||'remote'}`+(health.queue_length!=null?` · Hàng đợi: ${health.queue_length}`:'');
    if(banner)banner.hidden=true;
  }

  const msg=$('#gpuMsg');
  if(msg){
    if(isGpuOnline){
      msg.textContent='đang hoạt động ✓'+(gpu.demo?' (demo)':gpu.busy?' (bận)':'')+(gpu.url?' · '+gpu.url:'');
      msg.style.color='var(--ok)';
    }else{
      msg.textContent='ngoại tuyến ✗'+(gpu.error?' ('+gpu.error+')':'');
      msg.style.color='var(--warn)';
    }
  }
}

async function loadGpuConfig(){
  if(!API.isReady())return;
  const epoch=API._sessionEpoch;
  try{
    const cfg=await(await API.f('/api/config/gpu')).json();
    if(epoch!==API._sessionEpoch||!API.isReady())return;
    const urlInput=$('#gpuUrl');const tokInput=$('#gpuToken');
    if(urlInput&&document.activeElement!==urlInput){
      const nextVal=cfg.url||'';
      if(!urlInput.value||urlInput.value===nextVal||urlInput.value===(urlInput.dataset.syncedUrl||'')){
        urlInput.value=nextVal;
      }
      urlInput.dataset.syncedUrl=nextVal;
    }
    if(tokInput){
      tokInput.placeholder=cfg.has_token?'đã lưu (nhập mới để đổi)':'Mã token bảo mật GPU';
      tokInput.value='';
    }
  }catch{}
}

function syncGpuStatus(gpu){
  const g=gpu||{};
  const nextOnline=!!g.online;
  const nextUrl=(g.url||'').trim().replace(/\/+$/,'');
  const becameOnline=nextOnline&&(!API._gpuOnline||(!!nextUrl&&!!API._gpuUrl&&nextUrl!==API._gpuUrl));
  API._gpuOnline=nextOnline;
  if(nextUrl||!nextOnline)API._gpuUrl=nextUrl;
  return becameOnline;
}

let _sseConnecting=false;
let _sseEpoch=0;
async function connectGlobalSSE(){
  if(!API.isReady()||typeof EventSource==='undefined'){disconnectGlobalSSE();return}
  if(globalSSE||_sseConnecting||sseFallbackPolling)return;
  const epoch=++_sseEpoch;
  _sseConnecting=true;
  try{
    const url=await API.mediaUrl('/api/events');
    if(epoch!==_sseEpoch||!API.isReady()||sseFallbackPolling||globalSSE)return;
    const es=new EventSource(url);
    globalSSE=es;
    es.onopen=()=>{if(globalSSE===es)sseErrorCount=0};
    es.addEventListener('gpu_status',e=>{
      if(globalSSE!==es)return;
      try{
        const gpu=JSON.parse(e.data);
        updateChipAndGpuUI(gpu);
      }catch{}
    });
    es.addEventListener('job_update',e=>{
      if(globalSSE!==es)return;
      try{
        if(typeof loadJobs==='function')loadJobs();
      }catch{}
    });
    es.addEventListener('alert_reviewed',e=>{
      if(globalSSE!==es)return;
      try{const evData=JSON.parse(e.data);applyAlertReviewLive(evData)}catch{}
    });
    es.onerror=()=>{
      if(globalSSE!==es)return;
      sseErrorCount++;
      if(sseErrorCount>=3){
        sseFallbackPolling=true;
        disconnectGlobalSSE();
      }
    };
  }catch{if(epoch===_sseEpoch)globalSSE=null}
  finally{if(epoch===_sseEpoch)_sseConnecting=false}
}
function disconnectGlobalSSE(){
  _sseEpoch++;
  _sseConnecting=false;
  if(globalSSE){try{globalSSE.close()}catch{}globalSSE=null}
}
function updateChipAndGpuUI(gpu,forceSites=false){
  if(!API.isReady()){
    updateChipAndBanner(API.ok?{ok:true,authenticated:API.authenticated,auth_required:API.authRequired}:null);
    return null;
  }
  API._gpuSeq++;
  updateChipAndBanner({ok:true,authenticated:API.authenticated,auth_required:API.authRequired,gpu});
  const becameOnline=syncGpuStatus(gpu);
  const isOnline=!!(gpu&&gpu.online);
  if((becameOnline||((forceSites||API._sitesNeedRetry)&&isOnline))&&typeof loadSites==='function'){
    return loadSites();
  }
  return null;
}
function applyAlertReviewLive(evData){
  if(!evData)return;
  if(DATA&&DATA.events){
    const ev=DATA.events.find(x=>String(x.id)===String(evData.event_id)||String(x.alert_id)===String(evData.alert_id)||(S.jobId&&`${S.jobId}:${x.id}`===String(evData.alert_id)));
    if(ev)ev.review=evData.review;
  }
  if(GLOBAL_ALERTS&&GLOBAL_ALERTS.length){
    const item=GLOBAL_ALERTS.find(x=>x.alert_id===evData.alert_id||(x.job_id===evData.job_id&&String(x.event_id)===String(evData.event_id)));
    if(item)item.review=evData.review;
  }
  renderAlerts();
}

/* ---------- Nhật ký, trạng thái ---------- */
/* Bỏ dòng thanh tiến trình tải model (tqdm / huggingface: "Fetching 1 files: 100%|███| …") */
const noisyLog=l=>/\d+%\||it\/s|s\/it|^Fetching \d+ files|\[\d+:\d+<|[█▏▎▍▌▋▊▉]/.test(l);
/* Chỉ hiện tên bước đang chạy (không hiện log thô của pipeline như [DMA] ..., đường dẫn checkpoint) */
function stageLine(j,seen){const n=(j.stages||[]).length,i=j.stage;if(j.status==='running'&&n&&i>=0&&i<n&&!seen.has('s'+i)){seen.add('s'+i);logLine(`[${i+1}/${n}] ${j.stages[i]}`)}}
function logLine(m){const el=$('#srcmsg');el.textContent=m;el.className='srcmsg'+(/^(LỖI|Chưa|Camera chưa)/.test(m)?' err':'')}
function setStatus(t){$('#vstat').textContent=t}
function setBusy(b){S.running=b;$('#modeSel').classList.toggle('lock',b);$('#runBtn').disabled=b;$('#stopBtn').disabled=!b;$('#uploadBtn').disabled=b;$('#src').disabled=b;$('#camSel').disabled=b;$('#camUrl').disabled=b;if(b)$('#runBtn').textContent='Đang chạy…';else applySourceUI()}

/* ---------- System status ---------- */
let SYS=null;const sim={cpu:6,ram:2.4,gpu:0,vram:.3};
const ring=(pct,main,sub,label,col)=>`<div class="ring"><svg viewBox="0 0 80 80"><circle cx="40" cy="40" r="34" class="rbg"/><circle cx="40" cy="40" r="34" class="rfg" style="stroke:${col};stroke-dasharray:${(clamp(pct,0,100)/100*213.6).toFixed(1)} 213.6"/></svg><div class="rtxt"><b>${main}</b><small>${esc(sub)}</small></div><span>${label}</span></div>`;
function renderSystem(){
  if(!$('#rings'))return;                    // thẻ System status đã bỏ khỏi trang
  const c=SYS&&SYS.cpu,r=SYS&&SYS.ram,g=SYS&&SYS.gpus&&SYS.gpus[0],na='—';
  $('#rings').innerHTML=[
    ring(c?c.pct:0,c?c.pct.toFixed(1)+'%':na,c?'load '+c.load:'','CPU','#5B9BEA'),
    ring(r?r.used_gb/r.total_gb*100:0,r?r.used_gb+' GB':na,r?'/ '+r.total_gb+' GB':'','RAM','#7B6CE0'),
    ring(g?g.util:0,g?Math.round(g.util)+'%':na,g?g.name:'','GPU','#2E9B6A'),
    ring(g?g.mem_used_gb/g.mem_total_gb*100:0,g?g.mem_used_gb+' GB':na,g?'/ '+g.mem_total_gb+' GB':'','VRAM','#E0601C')].join('');
  $('#sysTag').textContent=SYS?(API.ok?'':'mô phỏng'):'chưa có dữ liệu'}
async function pollSystem(){
  if(!$('#rings'))return;
  if(API.isReady()){try{SYS=await API.system()}catch{SYS=null}}
  else if(CFG.demo){const j=(v,a,b,s)=>clamp(v+(Math.random()-.5)*s,a,b),busy=S.running;
    sim.cpu=j(sim.cpu,busy?40:2,busy?85:12,6);sim.ram=j(sim.ram,2,busy?9:3.5,.4);sim.gpu=j(sim.gpu,busy?30:0,busy?95:3,10);sim.vram=j(sim.vram,busy?4:.3,busy?12:.6,.5);
    SYS={cpu:{pct:+sim.cpu.toFixed(1),load:+(sim.cpu/25).toFixed(2)},ram:{used_gb:+sim.ram.toFixed(1),total_gb:33.7},gpus:[{name:'Tesla T4',util:sim.gpu,mem_used_gb:+sim.vram.toFixed(1),mem_total_gb:15}]}}
  else SYS=null;
  renderSystem()}

/* ---------- Hiển thị dữ liệu ---------- */
const LEVEL={stealing:['Cao','hi'],holding:['Trung bình','mid'],touching:['Thấp','lo']};
function listEvents(){if(!S.liveSim)return DATA.events;
  return DATA.events.filter(e=>e.start<=S.t).map(e=>e.end<=S.t?e:{...e,decision:'CANDIDATE',behavior:'uncertain',conf:null,reason:'Đang chạm (chờ VLM kết luận)',provisional:true})}
function renderKpis(){
  const ev=listEvents(),an=ev.filter(e=>e.decision==='ANOMALY').length,cand=ev.filter(e=>e.decision==='CANDIDATE').length;
  const pct=clamp(Math.round(S.progress),0,100);
  const srcText=S.srcName||'Chưa chọn nguồn';
  const evSub=ev.length?(cand?`${an} bất thường · ${cand} cần xác nhận`:`${an} bất thường trong phiên`):'Chưa ghi nhận sự kiện';
  let alertTag='<span class="kpi-tag idle">Chờ phân tích</span>',alertSub='Chưa có dữ liệu phiên';
  if(an){alertTag='<span class="kpi-tag">Cần kiểm duyệt</span>';alertSub='Hành vi xâm phạm hiện vật'}
  else if(S.running){alertTag='<span class="kpi-tag warn">Đang xử lý</span>';alertSub='Đang quét khung hình'}
  else if(S.loaded){alertTag='<span class="kpi-tag">Ổn định</span>';alertSub='Không có hành vi bất thường'}
  $('#kpis').innerHTML=[
    `<div class="kpi"><div class="kpi-head"><small>Nguồn video</small></div><b class="kpi-val sm${S.srcName?'':' muted'}" title="${esc(srcText)}">${esc(srcText)}</b><span class="kpi-sub">${S.mode==='rt'?'Chế độ trực tiếp (Realtime)':'Chế độ xử lý tệp video'}</span></div>`,
    `<div class="kpi"><div class="kpi-head"><small>Tiến độ xử lý</small></div><b class="kpi-val">${pct}%</b><div class="kpi-bar"><i style="width:${pct}%"></i></div></div>`,
    `<div class="kpi${an?' red':''}"><div class="kpi-head"><small>Cảnh báo bất thường</small>${alertTag}</div><b class="kpi-val">${an}</b><span class="kpi-sub">${alertSub}</span></div>`,
    `<div class="kpi"><div class="kpi-head"><small>Tổng sự kiện</small></div><b class="kpi-val">${ev.length}</b><span class="kpi-sub">${evSub}</span></div>`
  ].join('')}
function thumb(e){const c=document.createElement('canvas');c.width=240;c.height=150;const x=c.getContext('2d'),col=evColor(e);
  x.fillStyle='#18232b';x.fillRect(0,0,240,150);x.fillStyle='#1d2740';x.fillRect(0,0,240,40);
  x.fillStyle='#33434f';x.fillRect(130,70,50,60);x.fillStyle='#cbd5ee';x.beginPath();x.ellipse(155,64,15,17,0,0,7);x.fill();
  x.fillStyle='rgba(165,180,210,.4)';rr(x,70,40,42,95,14);x.fill();x.beginPath();x.arc(91,36,11,0,7);x.fill();
  x.strokeStyle=col;x.lineWidth=3;x.strokeRect(62,22,58,118);x.strokeStyle='#5B9BEA';x.strokeRect(122,44,66,92);
  x.strokeStyle=col;x.lineWidth=4;x.beginPath();x.moveTo(108,78);x.lineTo(138,76);x.stroke();return c.toDataURL()}
let _noImgCache='';
function noImg(){if(_noImgCache)return _noImgCache;const c=document.createElement('canvas');c.width=240;c.height=150;const x=c.getContext('2d');x.fillStyle='#141A20';x.fillRect(0,0,240,150);
  x.fillStyle='#9AA4AD';x.font='500 12.5px '+UI_FONT;x.textAlign='center';x.fillText('Không có ảnh bằng chứng',120,78);return (_noImgCache=c.toDataURL())}
function reviewBadge(rv){return rv==='ok'?'<span class="st ok">✓ Đã xác nhận</span>':'<span class="st no">Đã bỏ qua</span>'}
async function renderAlerts(){
  const box=$('#alerts');
  if(!box)return;

  if(S.alertScope==='global'){
    if(!API.ok){
      box.innerHTML='<p class="note">Chưa kết nối Backend. Mở Cài đặt để kết nối máy chủ và xem toàn bộ lịch sử cảnh báo.</p>';
      return;
    }
    if(API.authRequired&&!API.authenticated){
      box.innerHTML='<p class="note">Máy chủ Backend yêu cầu xác thực. Mở Cài đặt để nhập Mã truy cập Backend (API Key).</p>';
      return;
    }
    box.innerHTML='<p class="note">Đang tải lịch sử cảnh báo…</p>';
    try{
      let q='/api/alerts?limit=100';
      if(S.filter!=='all')q+=`&decision=${encodeURIComponent(S.filter)}`;
      const res=await(await API.f(q)).json();
      GLOBAL_ALERTS=res.items||[];
      if(!GLOBAL_ALERTS.length){
        box.innerHTML='<p class="note empty">Không có cảnh báo nào phù hợp bộ lọc.</p>';
        return;
      }
      const fallbackImg=noImg();
      const cards=await Promise.all(GLOBAL_ALERTS.map(async a=>{
        const lab=evLabel(a);const col=evColor(a);
        const imgUrl=a.img_url?(await API.mediaUrl(a.img_url)):fallbackImg;
        const startStr=a.start_s!=null?fmt(a.start_s):'0:00';
        const endStr=a.end_s!=null?fmt(a.end_s):'0:00';
        const confStr=a.confidence==null?'—':((a.confidence*100)|0)+'%';
        const jobLabel=a.job_name||a.job_id||'';
        const camLabel=a.cam?` · ${a.cam}`:'';
        const evText=(a.evidence||'').trim();
        const whyHtml=evText?`<div class="why">“${esc(evText)}”</div>`:'';
        const btn=a.review?reviewBadge(a.review)
          :`<button type="button" class="btn sm" data-global-rv="ok" data-alert-id="${esc(a.alert_id)}">Xác nhận</button><button type="button" class="btn sm" data-global-rv="no" data-alert-id="${esc(a.alert_id)}">Bỏ qua</button>`;
        return `<div class="al" data-alert-id="${esc(a.alert_id)}" style="--c:${col}"><img src="${imgUrl}" alt="" onerror="this.onerror=null;this.src=noImg()">
          <div><div class="hd"><span class="badge">${lab}</span><b>Visitor #${a.track_id!=null?a.track_id:'?'}</b><span class="meta">→ ${esc(a.artifact_id||'Hiện vật')}</span></div>
          <div class="meta">${startStr}–${endStr} · ${esc(jobLabel)}${esc(camLabel)} · VLM tin cậy ${confStr}</div>
          ${whyHtml}<div class="acts">${btn}</div></div></div>`;
      }));
      box.innerHTML=cards.join('');
    }catch(err){
      box.innerHTML=`<p class="note err">Lỗi tải cảnh báo: ${esc(err.message)}</p>`;
    }
    return;
  }

  // S.alertScope === 'current'
  const list=listEvents().filter(e=>S.filter==='all'||e.decision===S.filter);
  if(!list.length){box.innerHTML='<p class="note empty">Chưa có cảnh báo.</p>';return}
  box.innerHTML=list.map(e=>{const a=DATA.arts.find(a=>a.id===e.art)||{name:'Hiện vật '+e.art};
    const lab=e.provisional?'Đang chạm':evLabel(e),col=e.provisional?'#C98A00':evColor(e);
    if(!e.img)e.img=DATA.real?noImg():thumb(e);
    const reasonText=(e.reason||'').trim();
    const whyHtml=reasonText?`<div class="why">“${esc(reasonText)}”</div>`:'';
    const seekBtn=`<button type="button" class="btn sm" data-seek="${e.start}">Xem lại</button>`;
    const rvPart=e.review?reviewBadge(e.review)
      :(e.decision==='NORMAL'||e.provisional)?'':`<button type="button" class="btn sm" data-rv="ok" data-id="${e.id}">Xác nhận</button><button type="button" class="btn sm" data-rv="no" data-id="${e.id}">Bỏ qua</button>`;
    return`<div class="al" data-id="${e.id}" style="--c:${col}"><img src="${e.img}" data-seek="${e.start}" alt="" onerror="this.onerror=null;this.src=noImg()">
      <div><div class="hd"><span class="badge">${lab}</span><b>Visitor #${e.track}</b><span class="meta">→ ${esc(a.name)}</span></div>
      <div class="meta">${fmt(e.start)}–${fmt(e.end)} · VLM tin cậy ${e.conf==null?'—':((e.conf*100)|0)+'%'}</div>
      ${whyHtml}<div class="acts">${seekBtn}${rvPart}</div></div></div>`}).join('');
}
function renderTimeline(){const tl=$('#tl');tl.querySelectorAll('.seg').forEach(n=>n.remove());
  listEvents().forEach(e=>{const d=document.createElement('div');d.className='seg';d.style.cssText=`left:${e.start/D*100}%;width:${(e.end-e.start)/D*100}%;background:${evColor(e)}`;d.title=`#${e.track} ${evLabel(e)}`;d.dataset.seek=e.start;tl.append(d)});
  const o=DER.occ;if(DATA.real||!o.length||Math.max(...o)===0){$('#occ').innerHTML='';return}
  const m=Math.max(...o),pts=o.map((v,i)=>`${i/(o.length-1)*100},${100-v/m*70-4}`).join(' ');
  $('#occ').setAttribute('viewBox','0 0 100 100');$('#occ').innerHTML=`<polyline points="0,100 ${pts} 100,100" fill="rgba(25,35,43,.07)" stroke="rgba(25,35,43,.4)" stroke-width=".8" vector-effect="non-scaling-stroke"/>`}
function renderAll(){renderKpis();renderAlerts();renderTimeline();if(typeof renderAnalytics==='function')renderAnalytics()}

function hideLive(){LIVE=null;$('#liveimg').hidden=true;$('#rthud').hidden=true;const u=$('#liveimg').src;if(u&&u.startsWith('blob:'))URL.revokeObjectURL(u);$('#liveimg').removeAttribute('src')}
function loadData(data,videoURL,opts={}){
  hideLive();S.liveSim=false;
  DATA=data;D=data.meta.duration;derive();S.t=0;S.loaded=true;S.filter='all';
  if(videoURL&&videoURL!==S.videoURL){S.videoURL=videoURL;vid.src=videoURL}
  vid.hidden=!S.videoURL;$('#vmsg').hidden=true;
  document.body.classList.toggle('real',!!data.real);
  S.layers.art=data.real?!opts.liveVideo:false;S.layers.labels=true;
  $$('#layers [data-l="art"]').forEach(b=>b.classList.toggle('on',S.layers.art));
  $$('#layers [data-l="labels"]').forEach(b=>b.classList.toggle('on',S.layers.labels));
  $('#map').hidden=false;$('#mapNote').hidden=true;
  S.srcName=opts.name||(data.real?data.meta.camera_id:(S.file?S.file.name:'Dữ liệu mẫu'));
  $$('#filters .tg').forEach(b=>b.classList.toggle('on',b.dataset.f==='all'));
  setPlaying(false);renderAll()}
function resetData(){resetMap();hideLive();S.liveSim=false;DATA=EMPTY;D=60;derive();S.loaded=false;S.t=0;document.body.classList.remove('real');$('#map').hidden=false;$('#mapNote').hidden=true;renderAll()}

/* ---------- Map 2D thật (multicam-2d-floorplan-tracker qua backend): ảnh map + vị trí người / hiện vật ---------- */
const MAP0=()=>({ver:-1,img:null,people:[],arts:[],frames:null,keys:null,fps:25,shown:0,trail:{}});
let MAP=MAP0();
function resetMap(){MAP=MAP0();if(typeof SITEDATA!=='undefined')SITEDATA=null}
async function pollMap(id,alive,full){
  try{const d=await(await API.f(`/api/jobs/${id}/map/data${full?'?full=1':''}`)).json();
    if(!alive()&&!full)return;
    if(d.has_map&&d.version!==MAP.ver){const u=await API.blob(id,'map');if(u){const im=new Image();im.onload=()=>{MAP.img=im};im.src=u;MAP.ver=d.version}}
    MAP.people=d.people||[];MAP.arts=d.artifacts||[];MAP.fps=d.fps||MAP.fps;
    for(const p of MAP.people){const a=MAP.trail[p.id]||(MAP.trail[p.id]=[]);a.push([p.x,p.y]);if(a.length>8)a.shift()}
    if(full&&d.frames){MAP.frames=d.frames;MAP.keys=Object.keys(d.frames).map(Number).sort((a,b)=>a-b);MAP.heat=null;MAP.analytics=null;if(typeof renderAnalytics==='function')renderAnalytics()}
  }catch{}}
async function mapLoop(id,alive){while(alive()){await pollMap(id,alive,false);await sleep(1000)}}
function mapPeopleAt(t){            // phát lại: vị trí theo khung gần nhất không vượt quá thời điểm t
  if(!MAP.keys||!MAP.keys.length)return null;const f=Math.round(t*MAP.fps);let lo=0,hi=MAP.keys.length-1;
  if(f<MAP.keys[0])return[];
  while(lo<hi){const m=(lo+hi+1)>>1;MAP.keys[m]<=f?lo=m:hi=m-1}
  return MAP.frames[MAP.keys[lo]].map(p=>({id:p[0],x:p[1],y:p[2]}))}
/* một người trên map: cả đường đi trong suốt video + MỘT chấm ở vị trí cuối cùng của người đó. pts = [[px,py],...] theo điểm ảnh canvas */
function drawPersonPath(x,pts,col,label,ink,showPath){
  if(!pts.length)return;
  if(showPath&&pts.length>1){x.strokeStyle=col;x.globalAlpha=.85;x.lineWidth=2.5;x.lineJoin='round';x.lineCap='round';x.beginPath();
    pts.forEach((q,i)=>i?x.lineTo(q[0],q[1]):x.moveTo(q[0],q[1]));x.stroke();x.globalAlpha=1}
  const[px,py]=pts[pts.length-1];x.fillStyle=col;x.beginPath();x.arc(px,py,7,0,7);x.fill();x.strokeStyle='#0B0F13';x.lineWidth=1.5;x.stroke();
  if(label){x.fillStyle=ink;x.font='600 11px '+UI_FONT;x.textAlign='left';x.fillText(label,px+10,py+4)}}
function drawRealMap(c,t){
  const {x,w,h}=fit(c);if(w<30||h<30)return;x.clearRect(0,0,w,h);
  const im=MAP.img,pad=10,s=Math.min((w-pad*2)/im.naturalWidth,(h-pad*2)/im.naturalHeight),iw=im.naturalWidth*s,ih=im.naturalHeight*s,ox=(w-iw)/2,oy=(h-ih)/2;
  x.save();x.beginPath();x.roundRect(ox,oy,iw,ih,6);x.clip();x.drawImage(im,ox,oy,iw,ih);x.restore();
  x.strokeStyle='rgba(255,255,255,.25)';x.lineWidth=1;x.beginPath();x.roundRect(ox,oy,iw,ih,6);x.stroke();
  const X=v=>ox+v*iw,Y=v=>oy+v*ih;
  if(S.mlayers.zones)for(const a of MAP.arts){x.setLineDash([4,4]);x.strokeStyle='rgba(91,155,234,.6)';x.lineWidth=1.2;x.beginPath();x.arc(X(a.x),Y(a.y),iw*.06,0,7);x.stroke();x.setLineDash([])}
  x.textAlign='center';
  for(const a of MAP.arts){x.fillStyle='#1B64B8';x.fillRect(X(a.x)-6,Y(a.y)-6,12,12);x.fillStyle='#fff';x.font='600 11px '+UI_FONT;x.fillText('Hiện vật '+a.id,X(a.x),Y(a.y)-10)}
  const replay=!LIVE&&!!MAP.keys&&MAP.keys.length>0&&!!S.videoURL,all=S.mlayers.all;
  x.textAlign='left';
  const touchers=new Map();                                  // id người chạm -> màu theo hành vi nặng nhất
  if(replay){const rank={touching:1,holding:2,stealing:3};
    for(const e of DATA.events){if(e.decision!=='ANOMALY')continue;const o=touchers.get(e.track);
      if(!o||(rank[e.behavior]||3)>o.r)touchers.set(e.track,{r:rank[e.behavior]||3,col:BEHC[e.behavior]||BEHC.stealing})}}
  const others=replay?(mapPeopleAt(t)||[]):MAP.people;       // đang chạy trực tiếp: chỉ người đang chạm / bị cảnh báo
  const live=replay?[]:others.filter(p=>p.state==='contact'||p.state==='alert');
  for(const p of others){const hot=replay?touchers.has(p.id):live.includes(p);if(hot||!all)continue;
    x.fillStyle='#3DDC97';x.beginPath();x.arc(X(p.x),Y(p.y),4.5,0,7);x.fill();x.fillStyle='#fff';x.font='600 10px '+UI_FONT;x.fillText('#'+p.id,X(p.x)+8,Y(p.y)+3)}
  if(replay){
    for(const[id,o]of touchers){const pts=[];
      for(const k of MAP.keys){const q=MAP.frames[k].find(r=>r[0]===id);if(q)pts.push([X(q[1]),Y(q[2])])}
      drawPersonPath(x,pts,o.col,'#'+id,'#fff',S.mlayers.trails)}
    MAP.shown=touchers.size}
  else{
    for(const p of live){const col=p.state==='alert'?BEHC.stealing:BEHC.touching,tr=(MAP.trail[p.id]||[]).map(q=>[X(q[0]),Y(q[1])]);
      drawPersonPath(x,tr.length?tr:[[X(p.x),Y(p.y)]],col,'#'+p.id,'#fff',S.mlayers.trails)}
    MAP.shown=live.length}
  MAP.unit='người chạm'}
function drawMapPanel(t,c=$('#map')){   // true nếu đã vẽ map thật; false thì vẽ khung giữ chỗ như cũ
  if(typeof SITEDATA!=='undefined'&&SITEDATA){drawSiteMap(c,t);return true}
  if(MAP.img&&MAP.img.complete&&MAP.img.naturalWidth){
    drawRealMap(c,t);
    if(S.mapTab==='heat')drawRealHeat(c);
    else if(S.mapTab==='bottleneck'||S.mapTab==='flow'||S.mapTab==='deadzone')drawRealOverlay(c,S.mapTab);
    else setMapInfo('');
    return true}
  if(S.mapTab!=='events'){
    if(S.loaded&&!DATA.real&&DATA.tracks&&DATA.tracks.length){drawDemoOverlay(c,S.mapTab);return false}
    const {x,w,h}=fit(c);x.clearRect(0,0,w,h);x.fillStyle='#F7F8F6';x.fillRect(0,0,w,h);
    x.strokeStyle='rgba(18,21,24,.06)';x.lineWidth=1;
    for(let gx=24;gx<w;gx+=32){x.beginPath();x.moveTo(gx,0);x.lineTo(gx,h);x.stroke()}
    for(let gy=24;gy<h;gy+=32){x.beginPath();x.moveTo(0,gy);x.lineTo(w,gy);x.stroke()}
    x.fillStyle='#555E66';x.font='500 13px '+UI_FONT;x.textAlign='center';
    x.fillText('Chưa có dữ liệu quỹ đạo — phân tích video để xem lớp này.',w/2,h/2);MAP.shown=0;setMapInfo('');return false}
  drawMap(c,DATA.real?{live:false,t:0,...S.mlayers}:{live:S.loaded,t,...S.mlayers});setMapInfo('');return false}
let _infoId='mapInfo';function setMapInfo(s){const el=$('#'+_infoId);if(el&&el.textContent!==s)el.textContent=s}
/* 4 khung phân tích: vẽ lại cùng các hàm của map chính, mỗi khung một canvas và một dòng chú thích. Nội dung không đổi theo
   thời gian phát nên chỉ vẽ lại vài lần mỗi giây. */
const MAP_PANELS=[['heat','mapHeat','infoHeat'],['flow','mapFlow','infoFlow']];
let _panelsAt=0;
function drawMapPanels(t){
  const now=performance.now();if(now-_panelsAt<400)return;_panelsAt=now;
  const keep={tab:S.mapTab,shown:MAP.shown,unit:MAP.unit};
  for(const[tab,cid,iid]of MAP_PANELS){const c=$('#'+cid);if(!c||!c.clientWidth)continue;S.mapTab=tab;_infoId=iid;drawMapPanel(t,c)}
  S.mapTab=keep.tab;_infoId='mapInfo';MAP.shown=keep.shown;MAP.unit=keep.unit}

/* ---------- Phát lại ---------- */
function setPlaying(p){S.playing=p;$('#play').textContent=p?'⏸':'▶';if(S.videoURL)p?vid.play().catch(()=>{}):vid.pause()}
function seek(t){S.t=clamp(t,0,D);if(S.videoURL)vid.currentTime=S.t}
vid.addEventListener('loadedmetadata',()=>{if(!S.loaded&&vid.duration)D=vid.duration});
let last=performance.now();
function tick(now){
  try{
    const dt=(now-last)/1000;last=now;
    if(S.videoURL)S.t=vid.currentTime||0;else if(S.playing){S.t+=dt*S.speed;if(S.t>=D){S.t=D;setPlaying(false)}}
    paint();
  }finally{
    requestAnimationFrame(tick);
  }
}
function paint(){
  if($('#v-dash').hidden)return;
  const t=S.t,L=$('#live'),tl=$('#tl');
  const hasTimeline=S.loaded||!!S.videoURL||!!LIVE;
  if(tl){tl.setAttribute('aria-valuemax',String(Math.round(hasTimeline?D:0)));tl.setAttribute('aria-valuenow',String(Math.round(hasTimeline?t:0)))}
  if(LIVE){const {x,w,h}=fit($('#scene'));if(w>=30&&h>=30)x.clearRect(0,0,w,h);
    $('#ph').style.left=(D?t/D*100:0)+'%';$('#time').textContent=fmt(t)+' / '+fmt(D);L.hidden=false;
    if(LIVE.banner&&performance.now()<LIVE.banner.until){L.className='live '+LIVE.banner.cls;L.textContent=LIVE.banner.text}else{L.className='live grn';L.textContent='● LIVE'}
    {const m=drawMapPanel(t);$('#mapCount').textContent=(m?MAP.shown:0)+' '+(MAP.unit||'người');drawMapPanels(t)}
    return}
  if(S.loaded)drawScene(t);else{const {x,w,h}=fit($('#scene'));if(w>=30&&h>=30)x.clearRect(0,0,w,h)}
  const hasMap=drawMapPanel(t);
  $('#ph').style.left=(hasTimeline&&D?t/D*100:0)+'%';$('#time').textContent=hasTimeline?(fmt(t)+' / '+fmt(D)):'0:00 / 0:00';
  if(!S.loaded)L.hidden=true;else{L.hidden=false;
    const aev=activeEvents(t).filter(e=>e.decision!=='NORMAL'),an=aev.find(e=>e.decision==='ANOMALY'),ca=aev.find(e=>e.decision==='CANDIDATE');
    if(an){L.className='live red';L.textContent=`Visitor #${an.track}: ${evLabel(an)} — ${(DATA.arts.find(a=>a.id===an.art)||{name:'hiện vật '+an.art}).name}`}
    else if(ca){L.className='live yel';L.textContent=`Visitor #${ca.track} cần xác nhận`}
    else{L.className='live grn';L.textContent='● Bình thường'}}
  $('#mapCount').textContent=hasMap?MAP.shown+' '+(MAP.unit||'người'):DATA.real?'0 người':(S.loaded?activeAt(t).length:0)+' người';
  drawMapPanels(t);
  const act=S.loaded?activeEvents(t):[];
  $$('#alerts .al').forEach(el=>el.classList.toggle('act',act.some(e=>e.id==el.dataset.id&&e.decision!=='NORMAL')))}

/* ---------- Chạy phân tích ---------- */
const DEMO_STEPS=[['Khởi động & nạp model',['Nạp detector YOLOX','Nạp tracker ByteTrack-DMA-LTC']],['Tách hiện vật',['MM-Grounding-DINO: 4 hiện vật','SAM2.1 tạo mask xong']],
  ['Nạp VLM',['Qwen3-VL-8B Q4_K_M sẵn sàng']],['Phân tích video',['Tracking: 7 track','Phát hiện 5 sự kiện chạm','VLM: touching 0.91','VLM: holding 0.87','VLM: stealing 0.93','VLM: uncertain 0.52']],['Ghi kết quả',['Xuất result.json']]];
function stopped(){setStatus('Đã dừng');logLine('Đã dừng.')}
async function runDemo(){
  setStatus('Đang chạy (mô phỏng)');logLine('Chế độ demo: kết quả là dữ liệu mẫu.');
  const dur=S.videoURL&&vid.duration?Math.round(vid.duration)||60:60;
  for(let i=0;i<DEMO_STEPS.length;i++){const [name,subs]=DEMO_STEPS[i];logLine(`[${i+1}/${DEMO_STEPS.length}] ${name}`);
    for(let k=0;k<subs.length;k++){if(S.cancel)return stopped();logLine('  '+subs[k]);S.progress=(i+(k+1)/subs.length)/DEMO_STEPS.length*95;renderKpis();await sleep(450)}}
  if(S.cancel)return stopped();
  S.progress=100;loadData(makeMock(dur),S.videoURL,{name:S.file?S.file.name:'Dữ liệu mẫu'});setStatus('Hoàn tất (dữ liệu mẫu)');logLine('Hoàn tất.')}
async function runLive(src){
  const fd=new FormData();if(S.file)fd.append('video',S.file);else fd.append('source',src);
  fd.append('options',JSON.stringify({conf:+$('#conf').value}));
  setStatus('Đang gửi yêu cầu');logLine(S.file?'Tải video lên backend…':'Gửi đường dẫn / URL tới backend…');
  const resSubmit=await(await API.f('/api/jobs',{method:'POST',body:fd})).json();
  const id=resSubmit.job_id;S.jobId=id;const seen=new Set();let key=null;
  resetMap();mapLoop(id,()=>S.running&&S.jobId===id);

  let jobSSE=null;let lastJobData=null;
  if(typeof EventSource!=='undefined'){
    try{
      jobSSE=new EventSource(await API.mediaUrl('/api/jobs/'+id+'/events'));
      jobSSE.addEventListener('status',e=>{try{lastJobData=JSON.parse(e.data)}catch{}});
      jobSSE.addEventListener('job_update',e=>{try{lastJobData=JSON.parse(e.data)}catch{}});
      jobSSE.addEventListener('done',e=>{try{lastJobData=JSON.parse(e.data)}catch{}});
    }catch{}
  }

  try{
    for(;;){
      if(S.cancel){if(jobSSE)jobSSE.close();await API.cancel(id);return stopped()}
      let j=lastJobData;
      if(!j||(j.status!=='done'&&j.status!=='error'&&j.status!=='cancelled')){
        try{j=await API.job(id)}catch{}
      }
      if(j){
        if(j.status==='queued'){
          setStatus(`Hàng đợi (#${j.queue_position||1})`);
          logLine(`Đang xếp hàng (#${j.queue_position||1}) chờ GPU…`);
          S.progress=5;renderKpis();
        }else if(j.status==='running'){
          setStatus('Đang chạy');
          S.progress=(Math.max(0,j.stage)+(j.progress||0))/((j.stages&&j.stages.length)||5)*100;
          stageLine(j,seen);renderKpis();
        }else if(j.status==='syncing'){
          setStatus('Đang đồng bộ');
          logLine('Đang đồng bộ kết quả về máy chủ…');
          S.progress=98;renderKpis();
        }else if(j.status==='done'){
          key=j.key;break;
        }else if(j.status==='error'){
          throw new Error(j.error||'Xử lý thất bại');
        }else if(j.status==='cancelled'){
          return stopped();
        }
      }
      await sleep(1500);
    }
  }finally{
    if(jobSSE)jobSSE.close();
  }

  const res=await API.result(id);
  let savedAlerts=[];
  try{
    const alRes=await(await API.f(`/api/alerts?job_id=${id}&limit=500`)).json();
    savedAlerts=alRes.items||[];
  }catch{}
  const alertMap=new Map();
  savedAlerts.forEach(a=>{if(a.event_id!=null)alertMap.set(String(a.event_id),a)});

  const imgs={};
  await Promise.all((res.events||[]).filter(e=>e.task||alertMap.has(String(e.event_id))).map(async e=>{
    const a=alertMap.get(String(e.event_id));
    if(a&&a.img_url)imgs[e.event_id]=await API.mediaUrl(a.img_url);
    else if(e.task)imgs[e.event_id]=await API.blob(id,`files/${key}/vlm_inputs/${e.task}/sheet.jpg`);
  }));

  const videoURL=await API.mediaUrl(`/api/jobs/${id}/video`);
  S.progress=100;
  await pollMap(id,()=>true,true);
  const dataObj=fromResult(res,e=>imgs[e.event_id]);
  await Promise.all((dataObj.events||[]).map(async e=>{
    const a=alertMap.get(String(e.id));
    if(a&&a.review)e.review=a.review;
    if(a&&a.img_url&&!imgs[e.id])e.img=await API.mediaUrl(a.img_url);
  }));

  loadData(dataObj,videoURL,{liveVideo:true,name:S.file?S.file.name:(src.split(/[\/]/).pop()||undefined)});
  setStatus('Hoàn tất');logLine('Hoàn tất.');
  if(typeof loadJobs==='function')loadJobs();
  return true;
}

/* ---------- Realtime ---------- */
function prepareLiveView(name){
  S.videoURL=null;vid.removeAttribute('src');vid.hidden=true;$('#vmsg').hidden=true;
  $('#liveimg').hidden=false;$('#rthud').hidden=false;$('#rthud').textContent='LIVE · đang khởi động…';
  DATA={real:true,raw:null,meta:{duration:60,camera_id:name,width:960,height:540,nw:960},arts:[],tracks:[],events:[]};D=60;derive();
  S.loaded=true;S.t=0;S.filter='all';document.body.classList.add('real');S.srcName=name;
  $('#map').hidden=false;$('#mapNote').hidden=true;renderAll()}
function applyLive(le){
  const L=LIVE;if(!L)return;
  if(le.meta&&!L.meta){L.meta=le.meta;
    DATA.arts=(le.meta.artifacts||[]).map(a=>({id:a.id,name:a.label||('Hiện vật '+a.id),box:a.box,pedestal:!!a.pedestal}));
    D=le.meta.n_frames/le.meta.fps;DATA.meta.duration=D;derive()}
  for(const e of le.events||[]){L.after=Math.max(L.after,e.seq);
    if(e.kind==='contact'){
      DATA.events.push({id:'c'+e.seq,track:e.track,art:e.artifact,start:e.t_s,end:e.t_s+.5,behavior:'uncertain',decision:'CANDIDATE',conf:null,
        reason:'Đang chạm (chờ VLM kết luận)',provisional:true,review:null,img:null});
      logLine(`Chạm: Visitor #${e.track} → hiện vật ${e.artifact}`)}
    else if(e.kind==='vlm'){
      const art=(e.artifacts||[])[0],i=DATA.events.findIndex(x=>x.provisional&&x.track===e.track&&x.art===art);
      const row={id:'v'+e.seq,track:e.track,art,start:e.start_s!=null?e.start_s:(i>=0?DATA.events[i].start:S.t),end:e.end_s!=null?e.end_s:S.t,
        behavior:e.decision==='ANOMALY'?(e.behavior||'touching'):e.decision==='CANDIDATE'?'uncertain':'normal',decision:e.decision,conf:e.confidence,
        reason:e.evidence||(e.decision==='NORMAL'?'VLM đánh giá hành vi bình thường.':e.decision==='ANOMALY'?'VLM kết luận có hành vi bất thường.':'VLM chưa đủ chắc chắn.'),review:null,img:null};
      if(i>=0)DATA.events.splice(i,1,row);else DATA.events.push(row);
      logLine(`VLM: ${e.label||e.decision}${e.confidence==null?'':' '+e.confidence} (Visitor #${e.track})`);
      if(e.decision==='ANOMALY')L.banner={text:`Visitor #${e.track}: ${evLabel(row)} — ${(DATA.arts.find(a=>a.id===art)||{name:'hiện vật '+art}).name}`,cls:'red',until:performance.now()+8000}}}
  const st=le.status;
  if(st){S.t=st.t_s||0;S.progress=st.n_frames?Math.max(0,st.frame)/st.n_frames*100:0;
    $('#rthud').textContent=`LIVE · khung ${Math.max(st.frame,0)}/${st.n_frames} · ${st.live_fps} FPS · bỏ ${st.dropped} khung · VLM chờ ${st.vlm_pending==null?0:st.vlm_pending}`}
  renderAll()}
async function runRealtime(src,cam){
  const fd=new FormData();if(!cam&&S.file)fd.append('video',S.file);else fd.append('source',src);
  fd.append('options',JSON.stringify({conf:+$('#conf').value,realtime:true,camera:!!cam}));
  setStatus('Realtime: đang khởi động');logLine('Realtime: gửi yêu cầu tới backend…');
  const resSubmit=await(await API.f('/api/jobs',{method:'POST',body:fd})).json();
  const id=resSubmit.job_id;S.jobId=id;
  prepareLiveView(cam?camChoice().name:(S.file?S.file.name:src));LIVE={id,after:0,meta:null,banner:null};const mine=LIVE;resetMap();mapLoop(id,()=>LIVE===mine);

  let stop=false;const img=$('#liveimg');let mjpegFailed=false;
  img.src=await API.mediaUrl(`/api/jobs/${id}/live/stream`);
  img.onerror=()=>{
    if(!mjpegFailed&&!stop&&LIVE===mine){
      mjpegFailed=true;
      framesPoll();
    }
  };
  const framesPoll=async()=>{
    while(!stop&&LIVE===mine&&mjpegFailed){
      try{const u=await API.blob(id,'live');if(u&&LIVE===mine){const old=img.src;img.src=u;if(old&&old.startsWith('blob:'))URL.revokeObjectURL(old)}}catch{}
      await sleep(120);
    }
  };

  let jobSSE=null;let lastJobData=null;
  if(typeof EventSource!=='undefined'){
    try{
      jobSSE=new EventSource(await API.mediaUrl('/api/jobs/'+id+'/events'));
      jobSSE.addEventListener('status',e=>{try{lastJobData=JSON.parse(e.data)}catch{}});
      jobSSE.addEventListener('job_update',e=>{try{lastJobData=JSON.parse(e.data)}catch{}});
      jobSSE.addEventListener('live',e=>{try{const le=JSON.parse(e.data);if(le&&LIVE===mine)applyLive(le)}catch{}});
      jobSSE.addEventListener('done',e=>{try{lastJobData=JSON.parse(e.data)}catch{}});
    }catch{}
  }

  const seen=new Set();let key=null;
  try{
    for(;;){
      if(S.cancel){if(jobSSE)jobSSE.close();await API.cancel(id);stop=true;LIVE=null;return stopped()}
      let j=lastJobData;
      if(!j||(j.status!=='done'&&j.status!=='error'&&j.status!=='cancelled')){
        try{j=await API.job(id)}catch{}
      }
      if(j){
        if(j.status==='queued'){
          setStatus(`Hàng đợi (#${j.queue_position||1})`);
          logLine(`Đang xếp hàng (#${j.queue_position||1}) chờ GPU…`);
        }else if(j.status==='running'){
          stageLine(j,seen);
          if(j.stage>=3||j.status==='done')setStatus('Realtime · LIVE');
        }else if(j.status==='syncing'){
          setStatus('Đang đồng bộ');logLine('Đang đồng bộ kết quả về máy chủ…');
        }else if(j.status==='done'){
          key=j.key;break;
        }else if(j.status==='error'){
          throw new Error(j.error||'Xử lý thất bại');
        }else if(j.status==='cancelled'){
          stop=true;LIVE=null;return stopped();
        }
      }
      let le=null;try{le=await(await API.f(`/api/jobs/${id}/live/events?after=${mine.after}`)).json()}catch{}
      if(le&&LIVE===mine)applyLive(le);
      await sleep(1000);
    }
  }catch(e){stop=true;LIVE=null;throw e}
  finally{if(jobSSE)jobSSE.close()}

  stop=true;
  const res=await API.result(id);
  let savedAlerts=[];
  try{
    const alRes=await(await API.f(`/api/alerts?job_id=${id}&limit=500`)).json();
    savedAlerts=alRes.items||[];
  }catch{}
  const alertMap=new Map();
  savedAlerts.forEach(a=>{if(a.event_id!=null)alertMap.set(String(a.event_id),a)});

  const imgs={};
  await Promise.all((res.events||[]).filter(e=>e.task||alertMap.has(String(e.event_id))).map(async e=>{
    const a=alertMap.get(String(e.event_id));
    if(a&&a.img_url)imgs[e.event_id]=await API.mediaUrl(a.img_url);
    else if(e.task)imgs[e.event_id]=await API.blob(id,`files/${key}/vlm_inputs/${e.task}/sheet.jpg`);
  }));

  const videoURL=await API.mediaUrl(`/api/jobs/${id}/video`);
  S.progress=100;
  await pollMap(id,()=>true,true);
  const dataObj=fromResult(res,e=>imgs[e.event_id]);
  await Promise.all((dataObj.events||[]).map(async e=>{
    const a=alertMap.get(String(e.id));
    if(a&&a.review)e.review=a.review;
    if(a&&a.img_url&&!imgs[e.id])e.img=await API.mediaUrl(a.img_url);
  }));

  loadData(dataObj,videoURL,{liveVideo:false,name:S.file?S.file.name:(src.split(/[\/]/).pop()||undefined)});
  setStatus('Hoàn tất');logLine('Hoàn tất realtime: đã thay danh sách tạm bằng kết quả cuối.');
  if(typeof loadJobs==='function')loadJobs();
}
async function runDemoRealtime(cam){
  if(cam){S.videoURL=null;vid.removeAttribute('src')}
  setStatus(cam?'Camera (mô phỏng)':'Realtime (mô phỏng)');logLine('Chế độ demo: mô phỏng realtime, cảnh báo hiện dần theo thời gian video.');
  const dur=S.videoURL&&vid.duration?Math.round(vid.duration)||60:60;
  loadData(makeMock(dur),S.videoURL,{name:cam?camChoice().name:(S.file?S.file.name:'Dữ liệu mẫu')});
  S.liveSim=true;S.progress=0;S.t=0;if(S.videoURL)vid.currentTime=0;setPlaying(true);renderAll();
  while(S.t<D-.05&&!S.cancel){S.progress=S.t/D*100;renderAll();await sleep(500)}
  setPlaying(false);
  if(S.cancel){S.progress=S.t/D*100;renderKpis();return stopped()}
  S.liveSim=false;S.progress=100;renderAll();setStatus('Hoàn tất (mô phỏng)');logLine('Hoàn tất.')}
function setMode(m){S.mode=m;try{localStorage.setItem('ef_mode',m)}catch{}
  $$('#modeSel .tg').forEach(b=>b.classList.toggle('on',b.dataset.mode===m));
  applySourceUI();renderCalBox();renderKpis()}
$('#modeSel').onclick=e=>{const b=e.target.closest('[data-mode]');if(b&&!S.running)setMode(b.dataset.mode)};

async function startAnalysis(){
  if(S.running)return;
  const cam=camMode(),src=cam?camChoice().url:$('#src').value.trim();
  if(cam&&!src&&!CFG.demo){setStatus('Chưa có URL camera');logLine('Camera chưa có URL: vào Cài đặt → Danh sách Camera để nhập (vd. rtsp://...).');return}
  if(!cam&&!S.file&&!src){setStatus('Chưa có video');logLine('Chưa có video: bấm Tải video hoặc nhập đường dẫn / URL.');return}
  if(API.ok&&API.authRequired&&!API.authenticated){setStatus('Cần xác thực Backend');logLine('Chưa xác thực Backend: vào Cài đặt để nhập Mã truy cập Backend (API Key).');return}
  S.cancel=false;S.progress=0;setBusy(true);renderKpis();
  try{
    if(API.isReady()&&!cam&&S.mode==='batch'&&S.files.length>=1&&($('#siteSel').value||calOn()))await runSite(S.files,$('#siteSel').value||'__custom__');
    else if(API.isReady()&&!cam&&S.mode==='batch'&&S.files.length>1)await runMulti(S.files);
    else if(API.isReady())await(S.mode==='rt'?runRealtime(src,cam):runLive(src));
    else if(CFG.demo)await(S.mode==='rt'?runDemoRealtime(cam):runDemo());
    else{setStatus('Mất kết nối');logLine('Chưa kết nối Backend. Vào Cài đặt để nhập URL máy chủ, hoặc bật "Chạy trực tiếp trên máy chủ backend" nếu trang được mở từ chính server.')}
  }catch(err){logLine('LỖI: '+err.message);setStatus('Lỗi')}
  finally{setBusy(false);pollSystem()}}

/* ---------- Nhiều camera: mỗi video chạy riêng, mỗi camera một tab (video, cảnh báo, map riêng) ---------- */
S.files=[];let CAMRES=[],CAMIDX=-1;
function camSnap(name){return{name,data:DATA,videoURL:S.videoURL,map:MAP}}
function renderCamTabs(){const el=$('#camTabs');el.hidden=CAMRES.length<2;
  el.innerHTML=CAMRES.map((c,i)=>{const n=c.data.events.filter(e=>e.decision==='ANOMALY').length;
    return`<button type="button" class="tg${i===CAMIDX?' on':''}" data-cam="${i}" title="${esc(c.name)}">${esc(c.label||('CAM '+(i+1)))}${n?` · ${n}`:''}</button>`}).join('')}
function showCam(i){const c=CAMRES[i];if(!c)return;CAMIDX=i;MAP=c.map;
  loadData(c.data,c.videoURL,{liveVideo:true,name:c.name});S.srcName=c.name;renderKpis();renderCamTabs()}
$('#camTabs').onclick=e=>{const b=e.target.closest('[data-cam]');if(b&&!S.running)showCam(+b.dataset.cam)};
async function runMulti(files){
  CAMRES=[];CAMIDX=-1;renderCamTabs();
  for(let i=0;i<files.length;i++){
    if(S.cancel)break;
    S.file=files[i];S.srcName=files[i].name;S.progress=0;renderKpis();
    logLine(`[Camera ${i+1}/${files.length}] ${files[i].name}`);setStatus(`Camera ${i+1}/${files.length}`);
    const ok=await runLive('');
    if(!ok||S.cancel)break;
    CAMRES.push(camSnap(files[i].name));CAMIDX=CAMRES.length-1;renderCamTabs()}
  if(CAMRES.length){showCam(0);setStatus(CAMRES.length>1?`Hoàn tất ${CAMRES.length} camera`:'Hoàn tất');logLine(CAMRES.length>1?`Hoàn tất ${CAMRES.length} camera: bấm CAM 1, CAM 2... để xem từng camera.`:'Hoàn tất.')}}

/* ---------- Địa điểm đã hiệu chuẩn: ghép nhiều camera thành MỘT map (multicam-2d-floorplan-tracker) ---------- */
const SITE_COLORS=['#5B9BEA','#3DDC97','#FFA040','#B388FF','#FF6B8A','#4DD0E1','#FFD54F'];
let SITES=[],SITEDATA=null,_sitesReqId=0;
async function loadSites(){const s=$('#siteSel');if(!s)return;
  const reqId=++_sitesReqId,prevVal=s.value;
  if(!API.isReady()){
    API._sitesNeedRetry=false;
    SITES=[];
    s.innerHTML='<option value="">Địa điểm: không (mỗi camera một map)</option>';
    s.hidden=true;
    if(s.value!==prevVal&&typeof renderCalBox==='function')renderCalBox();
    return;
  }
  let next;
  try{next=await(await API.f('/api/sites')).json()}catch{
    if(reqId!==_sitesReqId)return;
    if(API.isReady())API._sitesNeedRetry=true;
    return;
  }
  if(reqId!==_sitesReqId||!API.isReady())return;
  API._sitesNeedRetry=false;
  SITES=Array.isArray(next)?next:[];
  s.innerHTML='<option value="">Địa điểm: không (mỗi camera một map)</option>'+SITES.map(x=>`<option value="${esc(x.site_id)}">${esc(x.name)}</option>`).join('');
  s.hidden=!SITES.length;
  let saved='';try{saved=localStorage.getItem('ef_site')||''}catch{}
  const pick=prevVal||saved;
  if(pick&&SITES.some(x=>x.site_id===pick))s.value=pick;
  if(s.value!==prevVal&&typeof renderCalBox==='function')renderCalBox()}
$('#siteSel').onchange=()=>{try{localStorage.setItem('ef_site',$('#siteSel').value)}catch{}renderCalBox()};
async function loadImg(path){const u=await blobUrl(path);if(!u)return null;return new Promise(r=>{const im=new Image();im.onload=()=>r(im);im.onerror=()=>r(null);im.src=u})}
async function blobUrl(path){try{return URL.createObjectURL(await(await API.f(path)).blob())}catch{return null}}
function assignCams(files,site){          // gán video vào camera: theo số trong tên file (cam1, C2...), còn lại theo thứ tự
  const free=[...site.cameras],out=[],rest=[];
  for(const f of files){const m=f.name.match(/(?:^|[^a-z0-9])c(?:am)?[ _-]?(\d+)/i),id=m&&'C'+m[1],i=id?free.indexOf(id):-1;
    if(i>=0){out.push({cam:id,file:f});free.splice(i,1)}else rest.push(f)}
  for(const f of rest){if(!free.length)break;out.push({cam:free.shift(),file:f})}
  return out.sort((a,b)=>site.cameras.indexOf(a.cam)-site.cameras.indexOf(b.cam))}
async function runSite(files,siteId){
  const custom=siteId==='__custom__';           // hiệu chuẩn tự nhập: camera đặt tên theo thứ tự video
  let site,map;
  if(custom){const ids=files.map((_,i)=>'C'+(i+1));
    site={site_id:siteId,name:'Hiệu chuẩn tự nhập',cameras:ids,labels:Object.fromEntries(files.map((f,i)=>[ids[i],`CAM ${i+1} · ${f.name}`]))};
    map=files.map((f,i)=>({cam:ids[i],file:f}))}
  else{site=SITES.find(s=>s.site_id===siteId);map=assignCams(files,site)}
  if(!map.length)throw new Error('Không gán được video vào camera của địa điểm này');
  if(files.length>map.length)logLine(`Địa điểm ${site.name} chỉ có ${site.cameras.length} camera: bỏ qua ${files.length-map.length} video thừa.`);
  CAMRES=[];CAMIDX=-1;renderCamTabs();resetMap();
  const fd=new FormData();fd.append('site',siteId);fd.append('events','1');
  if(custom){if(S.cal.all)fd.append('calib_all',S.cal.all);
    map.forEach((m,i)=>{fd.append('cam_'+i,m.file);fd.append('label_'+i,site.labels[m.cam]);if(S.cal.per[i])fd.append('calib_'+i,S.cal.per[i])})}
  else map.forEach(m=>fd.append(m.cam,m.file));
  logLine(map.map(m=>`${site.labels[m.cam]||m.cam} ← ${m.file.name}`).join(' · '));setStatus('Ghép camera');
  const id=(await(await API.f('/api/site-jobs',{method:'POST',body:fd})).json()).job_id;S.jobId=id;
  for(;;){
    if(S.cancel){logLine('Đã dừng. Server vẫn xử lý xong lượt ghép hiện tại.');return stopped()}
    const j=await(await API.f('/api/site-jobs/'+id)).json();
    if(j.status==='queued'){
      setStatus(`Hàng đợi (#${j.queue_position||1})`);
      logLine(`Đang xếp hàng (#${j.queue_position||1}) chờ GPU…`);
    }else if(j.status==='syncing'){
      setStatus('Đang đồng bộ');
      logLine('Đang đồng bộ kết quả ghép camera…');
    }else if(j.status==='running'){
      setStatus('Ghép camera');
      S.progress=(j.stage+.5)/(j.stages&&j.stages.length?j.stages.length:3)*100;renderKpis();
      if(j.stages&&j.stages[j.stage])logLine(`[${j.stage+1}/${j.stages.length}] ${j.stages[j.stage]}`);
    }
    if(j.status==='error')throw new Error(j.error);
    if(j.status==='done')break;
    await sleep(1500)}
  const res=await(await API.f(`/api/site-jobs/${id}/result`)).json();
  SITEDATA={bounds:res.bounds,cams:res.cameras,frames:res.frames,fps:res.fps,events:res.events||[],artifacts:res.artifacts||[],err:res.events_error||null,engine:res.engine||[],mapSource:res.map_source||'',mapNote:res.map_note||null};
  if(res.map)SITEDATA.mapImg=await loadImg(`/api/site-jobs/${id}/map`);                // mặt bằng của repo; null -> vẽ lưới như dự phòng
  SITEDATA.heat=computeSiteHeat();SITEDATA.view=computeSiteView();
  SITEDATA.analytics=EFAnalytics.computeMuseumAnalytics({heatmap:SITEDATA.heat,cams:SITEDATA.cams,exhibits:SITEDATA.artifacts});
  const dur=res.frames.length/(res.fps||1);
  for(const m of map){
    const evs=SITEDATA.events.filter(e=>e.cams.includes(m.cam)).map(e=>({id:e.id,track:e.person_id!=null?e.person_id:'?',art:e.artifact!=null?e.artifact:'?',start:e.t0,end:e.t1,
      behavior:e.behavior,decision:'ANOMALY',conf:e.conf,reason:e.evidence||'Engine phát hiện hành vi bất thường.',review:null,img:null}));
    const data={real:true,raw:null,meta:{duration:dur,camera_id:m.file.name,width:1920,height:1080,nw:960},arts:[],tracks:[],events:evs};
    const camVideoURL=await API.mediaUrl(`/api/site-jobs/${id}/video/${m.cam}`);
    CAMRES.push({name:m.file.name,label:site.labels[m.cam]||m.cam,cam:m.cam,data,videoURL:camVideoURL,map:MAP})}
  S.progress=100;
  showCam(0);setStatus(`Hoàn tất ${CAMRES.length} camera`);
  const ids=new Set(res.frames.flatMap(f=>f.people.map(p=>p.id)));
  const eng=SITEDATA.engine.map(e=>`${(site.labels[e.cam]||e.cam).split(' · ')[0]}: ${e.tracks} người theo dõi, ${e.contact_s}s chạm, ${e.alerts} bất thường`).join(' · ');
  logLine(`Hoàn tất: ${CAMRES.length} camera ghép thành một map chung, ${ids.size} người (ID chung)`+(res.events!==undefined?`, ${SITEDATA.events.length} hành vi bất thường.`:'.')
    +(eng&&!SITEDATA.events.length?` Engine thấy → ${eng}.`:'')+(SITEDATA.err?` Lưu ý: ${SITEDATA.err}`:'')+(res.map_source==='fallback'?` Lưu ý: repo không dựng được mặt bằng nên đang vẽ lưới${res.map_note?' ('+String(res.map_note).slice(0,90)+')':''}.`:''));
  if(typeof loadJobs==='function')loadJobs();
  return true}
function siteFrameAt(t){const F=SITEDATA.frames;let lo=0,hi=F.length-1;if(!F.length||t<F[0].t)return -1;
  while(lo<hi){const m=(lo+hi+1)>>1;F[m].t<=t?lo=m:hi=m-1}return lo}
/* nền map ghép: sàn, lưới 5 m, vùng nhìn thấy của từng camera (dùng chung cho tab hành vi và tab heatmap) */
function siteBase(c){
  const {x,w,h}=fit(c);if(w<30||h<30)return null;x.clearRect(0,0,w,h);const D_=SITEDATA,[x0,y0,x1,y1]=D_.view||D_.bounds,pad=12;
  const sc=Math.min((w-pad*2)/(x1-x0),(h-pad*2)/(y1-y0)),ox=(w-(x1-x0)*sc)/2,oy=(h-(y1-y0)*sc)/2,X=v=>ox+(v-x0)*sc,Y=v=>oy+(y1-v)*sc;
  if(D_.mapImg){                                  // mặt bằng do repo multicam-2d-floorplan-tracker dựng: vẽ người / hành vi lên chính ảnh này
    x.save();x.beginPath();x.roundRect(ox,oy,(x1-x0)*sc,(y1-y0)*sc,6);x.clip();x.drawImage(D_.mapImg,ox,oy,(x1-x0)*sc,(y1-y0)*sc);x.restore()}
  else{
    x.fillStyle='#FFFFFF';x.beginPath();x.roundRect(ox,oy,(x1-x0)*sc,(y1-y0)*sc,6);x.fill();
    x.strokeStyle='rgba(17,18,20,.08)';x.lineWidth=1;
    for(let g=Math.ceil(x0/5)*5;g<x1;g+=5){x.beginPath();x.moveTo(X(g),oy);x.lineTo(X(g),oy+(y1-y0)*sc);x.stroke()}
    for(let g=Math.ceil(y0/5)*5;g<y1;g+=5){x.beginPath();x.moveTo(ox,Y(g));x.lineTo(ox+(x1-x0)*sc,Y(g));x.stroke()}}
  return{x,w,h,X,Y,ox,oy,sc,D_,ink:'#111214'}}
function siteCams(B,heat){
  if(B.D_.mapImg)return;
  const {x,X,Y,D_,ox,oy,sc}=B,active=CAMRES[CAMIDX]&&CAMRES[CAMIDX].cam,[vx0,vy0,vx1,vy1]=D_.view||D_.bounds;
  x.save();x.beginPath();x.rect(ox,oy,(vx1-vx0)*sc,(vy1-vy0)*sc);x.clip();
  D_.cams.forEach((cm,i)=>{const col=SITE_COLORS[i%SITE_COLORS.length],on=cm.id===active;
    x.beginPath();cm.fov.forEach((p,k)=>k?x.lineTo(X(p[0]),Y(p[1])):x.moveTo(X(p[0]),Y(p[1])));x.closePath();
    if(!heat){x.globalAlpha=on?.28:.13;x.fillStyle=col;x.fill()}
    x.globalAlpha=on?.9:.45;x.strokeStyle=col;x.lineWidth=on?2:1;x.stroke();x.globalAlpha=1;
    x.fillStyle=col;x.beginPath();x.arc(X(cm.center[0]),Y(cm.center[1]),6,0,7);x.fill();
    x.fillStyle='#121518';x.font='600 11px '+UI_FONT;x.textAlign='left';x.fillText(cm.id,X(cm.center[0])+9,Y(cm.center[1])+4)});
  x.restore()}
const BEHC={stealing:'#FF4D4D',holding:'#FF7A1A',touching:'#FFB020'};
/* gộp sự kiện theo người: một người chạm nhiều lần vẫn chỉ một đường đi + một chấm, màu theo hành vi nặng nhất */
function sitePeopleTouched(events){
  const rank={touching:1,holding:2,stealing:3},by=new Map();
  for(const e of events){const key=e.person_id!=null?'p'+e.person_id:e.id,r=rank[e.behavior]||3,path=e.path&&e.path.length?e.path:[[e.t0,e.x,e.y]];
    const o=by.get(key);
    if(!o)by.set(key,{id:e.person_id,r,behavior:e.behavior,path});
    else{if(r>o.r){o.r=r;o.behavior=e.behavior}if(path.length>o.path.length)o.path=path}}
  return[...by.values()]}
/* map chỉ hiện NGƯỜI CHẠM hiện vật: cả đường đi trong suốt video + một chấm ở vị trí cuối */
function drawSiteMap(c,t){
  const B=siteBase(c);if(!B)return;const {x,w,h,X,Y,D_}=B;
  if(S.mapTab==='heat'){drawSiteHeat(B);return}
  if(S.mapTab==='bottleneck'||S.mapTab==='flow'||S.mapTab==='deadzone'){drawSiteOverlay(B,S.mapTab);return}
  siteCams(B,false);
  const EV=D_.events||[],who=sitePeopleTouched(EV),ids=new Set(who.map(o=>o.id));
  if(S.mlayers.all){const k=siteFrameAt(t);
    if(k>=0)for(const p of D_.frames[k].people){if(ids.has(p.id))continue;x.globalAlpha=D_.mapImg?.95:.55;x.fillStyle='#3DDC97';x.beginPath();x.arc(X(p.x),Y(p.y),D_.mapImg?5:4,0,7);x.fill();if(D_.mapImg){x.strokeStyle='#0B5D3B';x.lineWidth=1.5;x.stroke()}x.globalAlpha=.9;x.fillStyle=B.ink;x.font='600 10px '+UI_FONT;x.textAlign='left';x.fillText('#'+p.id,X(p.x)+7,Y(p.y)+3);x.globalAlpha=1}}
  for(const o of who)drawPersonPath(x,o.path.map(q=>[X(q[1]),Y(q[2])]),BEHC[o.behavior]||BEHC.stealing,o.id!=null?'#'+o.id:'',B.ink,S.mlayers.trails);
  if(!who.length){x.fillStyle='#555E66';x.font='500 13px '+UI_FONT;x.textAlign='center';
    x.fillText(D_.err?'Không phân tích được hành vi: '+D_.err.slice(0,70):'Chưa có người nào chạm hiện vật',w/2,h-18)}
  MAP.shown=who.length;MAP.unit='người chạm';
  setMapInfo(who.length?`${who.length} người chạm · ${EV.length} hành vi`:D_.err?'Chưa phân tích được hành vi':'Chưa thấy hành vi bất thường')}

/* ---------- Heatmap: phân tích mật độ, lượt khách & điểm dừng chân (xem js/heatmap.js) ---------- */
const HEATRAMP=EFHeat.HEATRAMP;
const heatColor=EFHeat.heatColor;
function heatLegend(x,w,h,modeData){const gx=16,gy=h-24,gw=152;const gr=x.createLinearGradient(gx,0,gx+gw,0);HEATRAMP.forEach(([p,c])=>gr.addColorStop(p,`rgb(${c})`));
  const label=modeData&&modeData.meta?modeData.meta.label:'Mật độ';
  x.fillStyle='rgba(18,21,24,.84)';rr(x,gx-8,gy-19,gw+16,34,5);x.fill();
  x.fillStyle=gr;rr(x,gx,gy,gw,6,3);x.fill();
  x.fillStyle='#F4F6F5';x.font='600 10px '+UI_FONT;x.textAlign='left';x.fillText('Ít',gx,gy-6);
  x.textAlign='center';x.fillStyle='#B5BEC6';x.fillText(label,gx+gw/2,gy-6);
  x.textAlign='right';x.fillStyle='#F4F6F5';x.fillText('Nhiều',gx+gw,gy-6)}
function heatPeakSummary(modeName,pk,coordText,cams,filteredCount){
  const parts=[`${EFHeat.MODE_META[modeName].peakLabel} ≈ ${coordText}`];
  if(cams&&cams.length)parts.push('vùng '+cams.join(', '));
  if(modeName==='visitors')parts.push(`${pk.uniqueVisitors} lượt khách · ~${Math.round(pk.presenceSec)}s`);
  else if(modeName==='dwell')parts.push(`${pk.uniqueVisitors} khách · ~${Math.round(pk.dwellSec)}s dừng (TB ${pk.avgDwellSec.toFixed(1)}s/khách)`);
  else parts.push(`${pk.uniqueVisitors} khách · ~${Math.round(pk.presenceSec)}s hiện diện`);
  if(filteredCount>0)parts.push(`đã lọc ${filteredCount} nhiễu tĩnh`);
  return parts.join(' · ')}
function drawHotspotMarkers(x,w,hotspots,toX,toY,ink){      // chỉ vòng đánh dấu, không ghi chữ trên bản đồ (chú thích nằm ở dòng dưới tab)
  hotspots.forEach((hs,idx)=>{
    const cx=toX(hs.x),cy=toY(hs.y),r=idx===0?13:9;
    x.globalAlpha=idx===0?1:.72;
    x.strokeStyle='rgba(255,255,255,.92)';x.lineWidth=idx===0?4:3;
    x.beginPath();x.arc(cx,cy,r,0,7);x.stroke();
    x.strokeStyle=ink||'#121518';x.lineWidth=idx===0?2:1.5;
    x.beginPath();x.arc(cx,cy,r,0,7);x.stroke();x.globalAlpha=1})}
function drawCanvasNotice(x,w,h,msg){
  x.save();x.font='500 12.5px '+UI_FONT;
  const tw=Math.min(w-32,x.measureText(msg).width+28),bh=34,bx=(w-tw)/2,by=(h-bh)/2;
  x.fillStyle='rgba(18,21,24,.84)';rr(x,bx,by,tw,bh,6);x.fill();
  x.fillStyle='#F4F6F5';x.textAlign='center';x.fillText(msg,w/2,by+21,tw-16);
  x.textAlign='left';x.restore()}
/* vùng tính heatmap / phân tích = phòng (bao các vùng nhìn camera), không tính lề và tiêu đề của ảnh mặt bằng */
function siteRoomBounds(){
  const pts=(SITEDATA.cams||[]).flatMap(c=>c.fov||[]);if(pts.length<3)return SITEDATA.bounds;
  let x0=1e9,y0=1e9,x1=-1e9,y1=-1e9;for(const[a,b]of pts){if(a<x0)x0=a;if(a>x1)x1=a;if(b<y0)y0=b;if(b>y1)y1=b}
  const B=SITEDATA.bounds,m=.3;return[Math.max(B[0],x0-m),Math.max(B[1],y0-m),Math.min(B[2],x1+m),Math.min(B[3],y1+m)]}
function computeSiteHeat(){return EFHeat.computeSiteHeat({...SITEDATA,bounds:siteRoomBounds()})}
function computeSiteView(){                   // khung nhìn thu gọn quanh người, camera và hành vi (không vẽ cả vùng đệm rộng của map)
  const D_=SITEDATA,B=D_.bounds;if(D_.mapImg)return B;let x0=1e9,y0=1e9,x1=-1e9,y1=-1e9,n=0;
  const add=(a,b)=>{if(a<x0)x0=a;if(a>x1)x1=a;if(b<y0)y0=b;if(b>y1)y1=b;n++};
  for(const f of D_.frames)for(const p of f.people)add(p.x,p.y);
  for(const e of D_.events||[]){add(e.x,e.y);for(const q of e.path||[])add(q[1],q[2])}
  for(const c of D_.cams)add(c.center[0],c.center[1]);
  if(!n)return B;
  const m=Math.max(3,.12*Math.max(x1-x0,y1-y0)),minS=14;
  x0-=m;x1+=m;y0-=m;y1+=m;
  if(x1-x0<minS){const c=(x0+x1)/2;x0=c-minS/2;x1=c+minS/2}
  if(y1-y0<minS){const c=(y0+y1)/2;y0=c-minS/2;y1=c+minS/2}
  return[Math.max(B[0],x0),Math.max(B[1],y0),Math.min(B[2],x1),Math.min(B[3],y1)]}
function pointInPoly(px,py,P){let ins=false;for(let i=0,j=P.length-1;i<P.length;j=i++){const[a,b]=P[i],[c,d]=P[j];if((b>py)!==(d>py)&&px<(c-a)*(py-b)/(d-b)+a)ins=!ins}return ins}
function heatEmptyMsg(H,modeName){
  const filt=H&&H.analytics&&H.analytics.filteredStaticIds?H.analytics.filteredStaticIds.length:0;
  if(H&&H.people===0&&filt>0)return `Đã lọc ${filt} track tĩnh đứng yên (không có khách di chuyển)`;
  return modeName==='dwell'?'Chưa ghi nhận điểm dừng chân đủ lâu':'Chưa có người nào được ghi nhận để vẽ heatmap'}
function drawSiteHeat(B){
  const {x,w,h,X,Y,ox,oy,sc,D_}=B,H=D_.heat,modeName=S.heatMode||'presence',M=H&&H.modes&&H.modes[modeName],[x0,y0]=(H&&H.bounds)||D_.bounds,[vx0,vy0,vx1,vy1]=D_.view||D_.bounds;
  if(!H||!M||!M.canvas||!M.peak){siteCams(B,true);x.fillStyle='#555E66';x.font='500 13px '+UI_FONT;x.textAlign='center';
    const msg=heatEmptyMsg(H,modeName);x.fillText(msg,w/2,h/2);
    MAP.shown=H?H.people:0;MAP.unit='người';setMapInfo(H&&H.people===0&&H.analytics.filteredStaticIds.length?msg:'');return}
  x.save();x.beginPath();x.roundRect(ox,oy,(vx1-vx0)*sc,(vy1-vy0)*sc,6);x.clip();x.imageSmoothingEnabled=true;
  x.drawImage(M.canvas,X(x0),Y(y0+H.ny*H.cell),H.nx*H.cell*sc,H.ny*H.cell*sc);x.restore();
  siteCams(B,true);
  drawHotspotMarkers(x,w,M.hotspots,X,Y,B.ink);
  heatLegend(x,w,h,M);
  const cams=D_.cams.filter(cm=>pointInPoly(M.peak.x,M.peak.y,cm.fov)).map(cm=>cm.id);
  MAP.shown=H.people;MAP.unit='người';
  setMapInfo(heatPeakSummary(modeName,M.peak,`(${M.peak.x.toFixed(1)}; ${M.peak.y.toFixed(1)}) m`,cams,H.analytics.filteredStaticIds.length))}
function drawRealHeat(c){                  // map 1 camera của dashboard (toạ độ 0-1 của ảnh map): heatmap từ vị trí theo khung khi phát lại
  const {x,w,h}=fit(c),im=MAP.img;if(w<30||h<30)return;
  if(!MAP.keys||!MAP.keys.length){x.fillStyle='rgba(12,15,18,.65)';x.fillRect(0,0,w,h);x.fillStyle='#B5BEC6';x.font='500 13px '+UI_FONT;x.textAlign='center';
    x.fillText('Bản đồ nhiệt hiển thị sau khi video chạy xong',w/2,h/2);MAP.shown=0;MAP.unit='người';setMapInfo('');return}
  if(!MAP.heat)MAP.heat=EFHeat.computeFrameHeat(MAP.frames,MAP.keys,MAP.fps);
  const pad=10,s=Math.min((w-pad*2)/im.naturalWidth,(h-pad*2)/im.naturalHeight),iw=im.naturalWidth*s,ih=im.naturalHeight*s,ox=(w-iw)/2,oy=(h-ih)/2;
  const H=MAP.heat,modeName=S.heatMode||'presence',M=H&&H.modes&&H.modes[modeName];
  MAP.shown=H?H.people:0;MAP.unit='người';
  if(!M||!M.canvas||!M.peak){const msg=heatEmptyMsg(H,modeName);drawCanvasNotice(x,w,h,msg);setMapInfo(msg);return}
  x.save();x.beginPath();x.roundRect(ox,oy,iw,ih,6);x.clip();x.fillStyle='rgba(8,11,14,.45)';x.fillRect(ox,oy,iw,ih);x.imageSmoothingEnabled=true;x.drawImage(M.canvas,ox,oy,iw,ih);x.restore();
  drawHotspotMarkers(x,w,M.hotspots,vx=>ox+vx*iw,vy=>oy+vy*ih);
  heatLegend(x,w,h,M);
  setMapInfo(heatPeakSummary(modeName,M.peak,`(${(M.peak.x*100).toFixed(0)}%, ${(M.peak.y*100).toFixed(0)}%)`,null,H.analytics.filteredStaticIds.length))}

/* ---------- Phân tích Không gian & Hành vi Khách (Proposals 1, 2, 4, 5, 7 — js/museum_analytics.js) ---------- */
function ensureDemoAnalytics(){
  if(!DATA||DATA.real||!S.loaded||!DATA.tracks||!DATA.tracks.length)return null;
  if(!DATA.heat){
    const dt=.5,frames=[];
    for(let t=0;t<=D+1e-6;t+=dt){
      const people=activeAt(t).map(p=>({id:p.id,x:p.fx,y:p.fy}));
      frames.push({t:+t.toFixed(3),people})}
    DATA.heat=EFHeat.computeHeatmap(frames,[0,0,1,1],{cell:1/48,sigma:.035,peakRadius:.06,yUp:false,isMetric:false,fps:2})}
  if(!DATA.analytics){
    DATA.analytics=EFAnalytics.computeMuseumAnalytics({heatmap:DATA.heat,exhibits:DATA.arts})}
  return{heat:DATA.heat,an:DATA.analytics,label:'Dữ liệu mẫu (demo)',isMetric:false}}
function ensureRealAnalytics(){
  if(!MAP.keys||!MAP.keys.length)return null;
  if(!MAP.heat)MAP.heat=EFHeat.computeFrameHeat(MAP.frames,MAP.keys,MAP.fps);
  if(!MAP.analytics)MAP.analytics=EFAnalytics.computeMuseumAnalytics({heatmap:MAP.heat,exhibits:MAP.arts});
  return{heat:MAP.heat,an:MAP.analytics,label:'Map 2D (1 camera)',isMetric:false}}
function currentAnalytics(){
  if(typeof EFAnalytics==='undefined')return null;
  if(typeof SITEDATA!=='undefined'&&SITEDATA&&SITEDATA.heat){
    if(!SITEDATA.analytics)SITEDATA.analytics=EFAnalytics.computeMuseumAnalytics({heatmap:SITEDATA.heat,cams:SITEDATA.cams,exhibits:SITEDATA.artifacts||[]});
    return{heat:SITEDATA.heat,an:SITEDATA.analytics,label:'Map ghép đa camera (m)',isMetric:true}}
  const r=ensureRealAnalytics();if(r)return r;
  return ensureDemoAnalytics()}

function drawArrowHead(x,x1,y1,x2,y2,sz){
  const ang=Math.atan2(y2-y1,x2-x1);
  x.beginPath();x.moveTo(x2,y2);
  x.lineTo(x2-sz*Math.cos(ang-Math.PI/6),y2-sz*Math.sin(ang-Math.PI/6));
  x.lineTo(x2-sz*Math.cos(ang+Math.PI/6),y2-sz*Math.sin(ang+Math.PI/6));
  x.closePath();x.fill()}

function drawOverlayLayer(x,w,h,tab,an,heat,toX,toY,drawGridCanvas,isMetric,isDark=false){
  const coordFmt=(px,py)=>isMetric?`(${px.toFixed(1)}; ${py.toFixed(1)}) m`:`(${(px*100).toFixed(0)}%, ${(py*100).toFixed(0)}%)`;
  if(tab==='bottleneck'){
    const bn=an.bottleneck;
    if(bn&&bn.canvas)drawGridCanvas(bn.canvas);
    const hs=(bn&&bn.hotspots)||[];
    hs.forEach((b,idx)=>{
      const px=toX(b.x),py=toY(b.y);
      x.strokeStyle=idx===0?'#FF4D79':'rgba(255,176,32,.9)';x.lineWidth=idx===0?2.5:1.8;
      x.beginPath();x.arc(px,py,idx===0?14:10,0,7);x.stroke();
      const txt=`Nghẽn #${idx+1} · ${b.peakConcurrent} khách (-${Math.round(b.speedDropPct*100)}% tốc độ)`;
      x.font='600 10.5px '+UI_FONT;x.textAlign='left';
      const tw=x.measureText(txt).width,lx=Math.max(8,Math.min(px+16,w-tw-10)),ly=Math.max(py-10,16);
      x.fillStyle='rgba(8,11,14,.84)';x.fillRect(lx-4,ly-11,tw+8,15);
      x.fillStyle=idx===0?'#FF8FA8':'#FFD480';x.fillText(txt,lx,ly)});
    MAP.shown=hs.length;MAP.unit='cổ chai';
    setMapInfo(hs.length?`${hs.length} nút thắt cổ chai · nặng nhất tại ${coordFmt(hs[0].x,hs[0].y)} (${hs[0].peakConcurrent} khách, giảm ${Math.round(hs[0].speedDropPct*100)}% tốc độ)`:'Không phát hiện nút thắt cổ chai (mật độ & tốc độ ổn định)');
    return}
  if(tab==='flow'){
    const fl=an.flow;
    // Zone-to-zone transition arcs
    const trs=(fl&&fl.transitions)||[];
    const maxCnt=trs.length?trs[0].count:1;
    trs.slice(0,8).forEach(tr=>{
      const x1=toX(tr.fromX),y1=toY(tr.fromY),x2=toX(tr.toX),y2=toY(tr.toY);
      x.strokeStyle=isDark?'rgba(255,148,77,.82)':'rgba(232,93,26,.72)';
      x.fillStyle=isDark?'#FF944D':'#E85D1A';
      x.lineWidth=1.75+2.8*(tr.count/maxCnt);
      x.beginPath();x.moveTo(x1,y1);x.lineTo(x2,y2);x.stroke();
      drawArrowHead(x,x1,y1,x2,y2,7.5)});
    // Dominant flow vectors
    const vecs=(fl&&fl.vectors)||[];
    const maxVis=vecs.reduce((m,v)=>Math.max(m,v.visitors),1);
    const vecRgb=isDark?'96,204,255':'27,100,184';
    vecs.forEach(v=>{
      const cx=toX(v.x),cy=toY(v.y);
      const ex=toX(v.x+v.ux*(fl.cellW*.38)),ey=toY(v.y+v.uy*(fl.cellH*.38));
      const alpha=(isDark?.55:.48)+.42*(v.visitors/maxVis);
      x.strokeStyle=`rgba(${vecRgb},${alpha.toFixed(2)})`;x.fillStyle=`rgba(${vecRgb},${alpha.toFixed(2)})`;
      x.lineWidth=1.5+1.5*(v.alignment||.5);
      x.beginPath();x.moveTo(cx,cy);x.lineTo(ex,ey);x.stroke();
      drawArrowHead(x,cx,cy,ex,ey,5.8)});
    MAP.shown=vecs.length;MAP.unit='vector luồng';
    setMapInfo(vecs.length?`Luồng di chuyển: ${vecs.length} vector chủ đạo`+(trs.length?` · mạnh nhất ${trs[0].fromName} → ${trs[0].toName} (${trs[0].count} lượt)`:''):'Chưa đủ bước di chuyển để vẽ luồng');
    return}
  if(tab==='deadzone'){
    const sp=an.space;
    if(sp&&sp.canvas)drawGridCanvas(sp.canvas);
    const cls=(sp&&sp.deadZoneClusters)||[];
    cls.slice(0,4).forEach((cl,idx)=>{
      const px=toX(cl.cx),py=toY(cl.cy);
      x.strokeStyle='#7AB8FF';x.lineWidth=1.6;x.setLineDash([4,3]);
      x.beginPath();x.arc(px,py,11,0,7);x.stroke();x.setLineDash([]);
      const pct=Math.round(cl.areaPctOfObserved*100);
      const txt=`Vùng chết #${idx+1} (${pct}%`+(cl.areaSqM!=null?` · ${cl.areaSqM.toFixed(1)} m²`:'')+')';
      x.font='600 10.5px '+UI_FONT;x.textAlign='left';
      const tw=x.measureText(txt).width,lx=Math.max(8,Math.min(px+13,w-tw-10)),ly=Math.max(py-8,16);
      x.fillStyle='rgba(8,11,14,.84)';x.fillRect(lx-4,ly-11,tw+8,15);
      x.fillStyle='#9EC8FF';x.fillText(txt,lx,ly)});
    MAP.shown=Math.round((sp?sp.activeAreaRatio:0)*100);MAP.unit='% khai thác';
    setMapInfo(sp?`Khai thác không gian: ${Math.round(sp.activeAreaRatio*100)}% hoạt động · ${Math.round(sp.deadAreaRatio*100)}% vùng chết (${cls.length} cụm lớn)`:'Chưa có dữ liệu không gian')}
}

function drawSiteOverlay(B,tab){
  const {x,w,h,X,Y,ox,oy,sc,D_}=B,H=D_.heat,[x0,y0]=H.bounds||D_.bounds,[vx0,vy0,vx1,vy1]=D_.view||D_.bounds;
  if(!D_.analytics)D_.analytics=EFAnalytics.computeMuseumAnalytics({heatmap:H,cams:D_.cams,exhibits:D_.artifacts||[]});
  const drawGrid=cv=>{x.save();x.beginPath();x.roundRect(ox,oy,(vx1-vx0)*sc,(vy1-vy0)*sc,6);x.clip();x.imageSmoothingEnabled=true;
    x.drawImage(cv,X(x0),Y(y0+H.ny*H.cell),H.nx*H.cell*sc,H.ny*H.cell*sc);x.restore()};
  drawOverlayLayer(x,w,h,tab,D_.analytics,H,X,Y,drawGrid,true);
  siteCams(B,true)}

function drawRealOverlay(c,tab){
  const {x,w,h}=fit(c),im=MAP.img;if(w<30||h<30)return;
  const cur=ensureRealAnalytics();
  if(!cur){x.fillStyle='rgba(12,15,18,.65)';x.fillRect(0,0,w,h);x.fillStyle='#B5BEC6';x.font='500 13px '+UI_FONT;x.textAlign='center';
    x.fillText('Lớp phân tích hiển thị sau khi video chạy xong',w/2,h/2);setMapInfo('');return}
  const pad=10,s=Math.min((w-pad*2)/im.naturalWidth,(h-pad*2)/im.naturalHeight),iw=im.naturalWidth*s,ih=im.naturalHeight*s,ox=(w-iw)/2,oy=(h-ih)/2;
  const drawGrid=cv=>{x.save();x.beginPath();x.roundRect(ox,oy,iw,ih,6);x.clip();x.fillStyle='rgba(8,11,14,.42)';x.fillRect(ox,oy,iw,ih);x.imageSmoothingEnabled=true;x.drawImage(cv,ox,oy,iw,ih);x.restore()};
  drawOverlayLayer(x,w,h,tab,cur.an,cur.heat,vx=>ox+vx*iw,vy=>oy+vy*ih,drawGrid,false,true)}

function drawDemoOverlay(c,tab){
  const cur=ensureDemoAnalytics();
  if(!cur)return;
  const {x,w,h}=fit(c);if(w<30||h<30)return;
  drawMap(c,{live:false,t:0,trails:false,zones:true,heat:false,all:false});
  const R=floorRect(w,h),toX=fx=>R.x+fx*R.w,toY=fy=>R.y+fy*R.h;
  const drawGrid=cv=>{x.save();x.beginPath();x.roundRect(R.x,R.y,R.w,R.h,6);x.clip();x.imageSmoothingEnabled=true;x.drawImage(cv,R.x,R.y,R.w,R.h);x.restore()};
  if(tab==='heat'){
    const modeName=S.heatMode||'presence',M=cur.heat.modes[modeName];
    if(M&&M.canvas&&M.peak){drawGrid(M.canvas);drawHotspotMarkers(x,w,M.hotspots,toX,toY);heatLegend(x,w,h,M);
      setMapInfo(heatPeakSummary(modeName,M.peak,`(${(M.peak.x*100).toFixed(0)}%, ${(M.peak.y*100).toFixed(0)}%)`,null,cur.heat.analytics.filteredStaticIds.length))}
    else{const msg=heatEmptyMsg(cur.heat,modeName);drawCanvasNotice(x,w,h,msg);setMapInfo(msg)}
    return}
  drawOverlayLayer(x,w,h,tab,cur.an,cur.heat,toX,toY,drawGrid,false)}

function drawAttractionMatrix(an){
  const c=$('#anMatrix');if(!c)return;
  const {x,w,h}=fit(c);if(w<10||h<10)return;
  x.clearRect(0,0,w,h);
  const padL=38,padR=16,padT=16,padB=28,pw=Math.max(40,w-padL-padR),ph=Math.max(40,h-padT-padB);
  const items=(an&&an.attraction&&an.attraction.items)||[];
  const thrA=an&&an.attraction?an.attraction.thresholds.attractionPower:.35;
  const thrH=an&&an.attraction?an.attraction.thresholds.holdingPowerSec:10;
  const maxH=Math.max(thrH*2.2,...items.map(it=>it.holdingPowerSec*1.15),20);
  const X=a=>padL+clamp(a,0,1)*pw,Y=s=>padT+(1-clamp(s/maxH,0,1))*ph;
  const tx=X(thrA),ty=Y(thrH);
  // 4 shaded quadrants
  x.fillStyle='rgba(61,220,151,.08)';x.fillRect(tx,padT,padL+pw-tx,ty-padT);         // Ngôi sao
  x.fillStyle='rgba(255,176,32,.07)';x.fillRect(tx,ty,padL+pw-tx,padT+ph-ty);        // Lướt nhanh
  x.fillStyle='rgba(91,155,234,.08)';x.fillRect(padL,padT,tx-padL,ty-padT);          // Viên ngọc ẩn
  x.fillStyle='rgba(148,160,172,.05)';x.fillRect(padL,ty,tx-padL,padT+ph-ty);        // Vùng lạnh
  // Axes & thresholds
  x.strokeStyle='rgba(17,18,20,.18)';x.lineWidth=1;x.strokeRect(padL,padT,pw,ph);
  x.setLineDash([4,4]);x.strokeStyle='rgba(17,18,20,.32)';
  x.beginPath();x.moveTo(tx,padT);x.lineTo(tx,padT+ph);x.moveTo(padL,ty);x.lineTo(padL+pw,ty);x.stroke();x.setLineDash([]);
  // Quadrant labels
  x.font='600 10px '+UI_FONT;x.fillStyle='#157347';x.textAlign='right';x.fillText('Ngôi sao',padL+pw-5,padT+12);
  x.fillStyle='#9A6000';x.fillText('Lướt nhanh',padL+pw-5,padT+ph-5);
  x.fillStyle='#1B64B8';x.textAlign='left';x.fillText('Viên ngọc ẩn',padL+5,padT+12);
  x.fillStyle='#555E66';x.fillText('Vùng lạnh',padL+5,padT+ph-5);
  // Axis captions
  x.fillStyle='#555E66';x.font='10px '+UI_FONT;x.textAlign='center';
  x.fillText('Attraction Power (% khách dừng ≥ 3s)',padL+pw/2,h-7);
  x.save();x.translate(12,padT+ph/2);x.rotate(-Math.PI/2);x.fillText('Holding (s)',0,0);x.restore();
  // Plot items
  if(!items.length){x.fillStyle='#555E66';x.font='12px '+UI_FONT;x.textAlign='center';x.fillText('Chưa có dữ liệu hiện vật / vùng',padL+pw/2,padT+ph/2);return}
  const placed=[];
  items.forEach(it=>{
    const px=X(it.attractionPower),py=Y(it.holdingPowerSec),col=it.quadrantMeta.color;
    x.fillStyle=col;x.beginPath();x.arc(px,py,6,0,7);x.fill();
    x.strokeStyle='#FFFFFF';x.lineWidth=1.5;x.stroke();
    x.fillStyle='#121518';x.font='600 10.5px '+UI_FONT;
    const rightSide=px<=padL+pw*.72;
    const tw=x.measureText(it.name).width;
    const lx=rightSide?Math.min(px+9,padL+pw-tw-2):Math.max(px-9,padL+tw+2);
    let ly=clamp(py+3.5,padT+12,padT+ph-4);
    for(let tries=0;tries<6;tries++){
      const hit=placed.some(b=>Math.abs(b.y-ly)<12&&((rightSide?lx:lx-tw)<b.x2+4)&&((rightSide?lx+tw:lx)>b.x1-4));
      if(!hit)break;
      ly=clamp(ly+(tries%2===0?-13:13)*(Math.floor(tries/2)+1),padT+12,padT+ph-4)}
    placed.push({x1:rightSide?lx:lx-tw,x2:rightSide?lx+tw:lx,y:ly});
    x.textAlign=rightSide?'left':'right';
    x.fillText(it.name,lx,ly)})}

function drawFatigueChart(an){
  const c=$('#anFatigueChart');if(!c)return;
  const {x,w,h}=fit(c);if(w<10||h<10)return;
  x.clearRect(0,0,w,h);
  const ft=an&&an.fatigue,qs=(ft&&ft.quartiles)||[];
  const padL=34,padR=34,padT=18,padB=26,pw=Math.max(40,w-padL-padR),ph=Math.max(40,h-padT-padB);
  x.strokeStyle='rgba(17,18,20,.14)';x.lineWidth=1;x.strokeRect(padL,padT,pw,ph);
  if(!ft||!ft.eligibleVisitors){
    x.fillStyle='#555E66';x.font='12px '+UI_FONT;x.textAlign='center';
    x.fillText(`Cần khách tham quan ≥ ${ft?ft.minTrackSec:60} giây để vẽ đường cong mỏi bảo tàng`,w/2,h/2);return}
  const maxSpd=Math.max(...qs.map(q=>q.avgSpeed),.1)*1.25;
  const bw=pw/4;
  // Bars = stopRatio (0..100%)
  qs.forEach((q,i)=>{
    const cx=padL+bw*(i+.5),barW=Math.min(34,bw*.45),bh=clamp(q.stopRatio,0,1)*ph;
    const isOnset=ft.onsetQuartile===q.quartile;
    x.fillStyle=isOnset?'rgba(255,176,32,.48)':'rgba(61,220,151,.35)';
    x.fillRect(cx-barW/2,padT+ph-bh,barW,bh);
    x.strokeStyle=isOnset?'#FFB020':'#3DDC97';x.lineWidth=1.2;
    x.strokeRect(cx-barW/2,padT+ph-bh,barW,bh);
    x.fillStyle='#3A4048';x.font='600 10.5px '+UI_FONT;x.textAlign='center';
    x.fillText('Q'+q.quartile,cx,h-8);
    x.fillStyle='#157347';x.font='10px '+UI_FONT;
    x.fillText(Math.round(q.stopRatio*100)+'%',cx,Math.max(padT+10,padT+ph-bh-4))});
  // Line = avgSpeed
  x.strokeStyle='#E85D1A';x.lineWidth=2.2;x.beginPath();
  qs.forEach((q,i)=>{const cx=padL+bw*(i+.5),cy=padT+(1-q.avgSpeed/maxSpd)*ph;i?x.lineTo(cx,cy):x.moveTo(cx,cy)});x.stroke();
  qs.forEach((q,i)=>{const cx=padL+bw*(i+.5),cy=padT+(1-q.avgSpeed/maxSpd)*ph;
    x.fillStyle='#E85D1A';x.beginPath();x.arc(cx,cy,4,0,7);x.fill();
    x.fillStyle='#C4490E';x.font='10px '+UI_FONT;x.textAlign='center';x.fillText(q.avgSpeed.toFixed(2),cx,Math.max(padT+10,cy-7))});
  // Mini legend
  x.font='10px '+UI_FONT;x.textAlign='left';x.fillStyle='#157347';x.fillText('■ Tỷ lệ dừng (%)',padL+4,padT-5);
  x.textAlign='right';x.fillStyle='#E85D1A';x.fillText('● Tốc độ TB',padL+pw-4,padT-5)}

function renderAnalytics(){
  if(!$('#analyticsSec'))return;
  const cur=currentAnalytics();
  const expBtn=$('#anExportBtn');
  if(!cur||!cur.an){
    $('#anMeta').textContent='Chưa có dữ liệu quỹ đạo';
    if(expBtn)expBtn.disabled=true;
    $('#anAttrTag').textContent='Attraction vs. Holding';
    $('#anAttrList').innerHTML='<p class="note empty">Chạy phân tích video để tính Attraction Power &amp; Holding Power.</p>';
    $('#anArchTag').textContent='0 khách';
    $('#anArchetypes').innerHTML='<p class="note empty">Chưa có dữ liệu khách tham quan.</p>';
    $('#anFatigueBadge').className='an-banner';
    $('#anFatigueBadge').textContent='Chưa có dữ liệu để đánh giá điểm mỏi bảo tàng & tỷ lệ bỏ qua.';
    $('#anTransitions').innerHTML='';
    drawAttractionMatrix(null);drawFatigueChart(null);return}

  const {an,label,isMetric}=cur;
  if(expBtn)expBtn.disabled=false;
  $('#anMeta').textContent=`${label} · ${an.meta.totalVisitors} khách ẩn danh · ${Math.round(an.meta.durationSec)}s`;

  // 1. Attraction vs Holding Matrix
  const at=an.attraction;
  $('#anAttrTag').textContent=`${at.source==='exhibits'?'Hiện vật':'Vùng tự động'} · TB ${Math.round(at.summary.avgAttractionPower*100)}% / ${at.summary.avgHoldingPowerSec.toFixed(1)}s`;
  drawAttractionMatrix(an);
  $('#anAttrList').innerHTML=at.items.length?at.items.map(it=>{
    const q=it.quadrantMeta;
    return`<div class="an-row" title="${esc(q.advice)}"><b title="${esc(it.name)}">${esc(it.name)}</b><span class="num">Hút ${Math.round(it.attractionPower*100)}% (${it.stoppers}/${it.passers})</span><span class="num">Giữ ${it.holdingPowerSec.toFixed(1)}s</span><span class="an-q" style="color:${q.color};border-color:${q.color}55;background:${q.color}18">${esc(q.label)}</span></div>`}).join('')
    :'<p class="note empty">Không có hiện vật/vùng nào được ghi nhận.</p>';

  // 7. Visitor Archetypes (Verón & Levasseur)
  const ar=an.archetypes;
  $('#anArchTag').textContent=`${ar.total} khách`+(ar.dominantMeta?` · Chủ đạo: ${ar.dominantMeta.label}`:'');
  $('#anArchetypes').innerHTML=['ant','butterfly','fish','grasshopper'].map(key=>{
    const m=ar.meta[key],cnt=ar.counts[key],pct=Math.round((ar.shares[key]||0)*100);
    const sub=ar.visitors.filter(v=>v.archetype===key);
    const avgStop=sub.length?Math.round(sub.reduce((s,v)=>s+v.stopRatio,0)/sub.length*100):0;
    const avgCov=sub.length?Math.round(sub.reduce((s,v)=>s+v.coverage,0)/sub.length*100):0;
    return`<div class="an-arch-item"><div class="an-arch-top"><b><i class="an-arch-dot" style="background:${m.color}"></i>${esc(m.label)}</b><span>${cnt} khách (${pct}%)</span></div><div class="an-bar"><i style="width:${pct}%;background:${m.color}"></i></div><div class="an-arch-desc"><span>${esc(m.desc)}</span><span>${sub.length?`Dừng ${avgStop}% · Phủ ${avgCov}%`:'—'}</span></div></div>`}).join('');

  // 4. Museum Fatigue + Skip Rate + Transitions
  const ft=an.fatigue,sk=an.skipRate,fl=an.flow;
  $('#anFatigueTag').textContent=`Bỏ qua TB ${Math.round(sk.avgPasserSkipRate*100)}%`;
  const fb=$('#anFatigueBadge');
  if(!ft.eligibleVisitors){fb.className='an-banner';fb.textContent=`Chưa đủ khách tham quan dài (≥ ${ft.minTrackSec}s) để phát hiện điểm mỏi bảo tàng.`}
  else if(ft.detected){fb.className='an-banner warn';fb.textContent=`Phát hiện dấu hiệu mỏi bảo tàng từ Q${ft.onsetQuartile||4} (~${ft.onsetProgressPct||75}% hành trình${ft.onsetAvgTimeSec!=null?', ~'+Math.round(ft.onsetAvgTimeSec)+'s':''}): tốc độ tăng ${Math.round(ft.speedIncreasePct*100)}%, tỷ lệ dừng giảm ${Math.round(ft.stopDropPct*100)}%.`}
  else{fb.className='an-banner ok';fb.textContent=`Nhịp tham quan ổn định trên ${ft.eligibleVisitors} khách (TB ${Math.round(ft.avgEligibleVisitSec)}s/khách) · Tỷ lệ bỏ qua hiện vật TB ${Math.round(sk.avgPasserSkipRate*100)}%.`}
  drawFatigueChart(an);
  $('#anTransitions').innerHTML=fl.transitions.length
    ?fl.transitions.slice(0,5).map(t=>`<span class="an-chip">${esc(t.fromName)} → ${esc(t.toName)} (${t.count})</span>`).join('')
    :'<span class="note" style="margin:0">Chưa ghi nhận luồng chuyển tiếp giữa các vùng.</span>';}

function redrawAnalyticsCharts(){
  const cur=currentAnalytics();
  drawAttractionMatrix(cur&&cur.an?cur.an:null);
  drawFatigueChart(cur&&cur.an?cur.an:null)}
let _anResizeTimer=0;
window.addEventListener('resize',()=>{clearTimeout(_anResizeTimer);_anResizeTimer=setTimeout(()=>{if(!$('#v-dash').hidden)redrawAnalyticsCharts()},120)});

$$('[data-mapjump]').forEach(btn=>{btn.onclick=()=>{const m=$(btn.dataset.mapjump==='deadzone'?'#panel-mapDead':'#panel-mapBottleneck');if(m)m.scrollIntoView({behavior:'smooth',block:'center'})}});
if($('#anExportBtn'))$('#anExportBtn').onclick=()=>{
  const cur=currentAnalytics();if(!cur||!cur.an)return;
  const clean=JSON.parse(JSON.stringify(cur.an,(k,v)=>{
    if(k==='grid'||k==='norm'||k==='deadMask'||k==='observedMask'||k==='peakConcurrentGrid'||k==='avgSpeedGrid'||k==='canvas')return undefined;
    return v}));
  const a=document.createElement('a');
  const url=URL.createObjectURL(new Blob([JSON.stringify(clean,null,2)],{type:'application/json'}));
  a.href=url;a.download='exhibitflow_museum_analytics.json';a.click();
  setTimeout(()=>URL.revokeObjectURL(url),1000)};

/* ---------- Hiệu chuẩn camera tự nhập: mỗi video một file JSON (K, R, t), hoặc một file chung cho cả các camera ---------- */
S.cal={per:{},all:null};
const calOn=()=>!!S.cal.all||Object.keys(S.cal.per).length>0;
function renderCalBox(){const box=$('#calBox'),preset=!!$('#siteSel').value;
  const show=S.mode==='batch'&&S.files.length>=1;box.hidden=!show;if(!show)return;
  $('#calRows').hidden=preset;$('#calAllRow').hidden=preset;
  $('#calRows').innerHTML=S.files.map((f,i)=>`<div class="calrow"><span title="${esc(f.name)}">CAM ${i+1} · ${esc(f.name)}</span><input type="file" accept=".json,application/json" data-cal="${i}" hidden><button class="btn sm" type="button" data-calbtn="${i}">Chọn hiệu chuẩn</button><em class="${S.cal.per[i]?'ok':''}">${S.cal.per[i]?esc(S.cal.per[i].name):'chưa chọn'}</em></div>`).join('');
  const a=$('#calAllName');a.textContent=S.cal.all?S.cal.all.name:'chưa chọn';a.className=S.cal.all?'ok':''}
$('#calRows').onclick=e=>{const b=e.target.closest('[data-calbtn]');if(b)$(`#calRows [data-cal="${b.dataset.calbtn}"]`).click()};
$('#calRows').onchange=e=>{const i=e.target.dataset.cal;if(i==null)return;if(e.target.files[0])S.cal.per[i]=e.target.files[0];else delete S.cal.per[i];renderCalBox()};
$('#calAllBtn').onclick=()=>$('#calAll').click();
$('#calAll').onchange=e=>{S.cal.all=e.target.files[0]||null;renderCalBox()};

/* ---------- Tab dưới map: hành vi | heatmap | cổ chai | luồng đi | vùng chết ---------- */
const _hm=$('#heatModes');if(_hm)_hm.onclick=e=>{const b=e.target.closest('[data-hm]');if(!b)return;S.heatMode=b.dataset.hm;$$('#heatModes .tg').forEach(x=>x.classList.toggle('on',x===b));_panelsAt=0};

/* ---------- Upload ---------- */
function setFile(f){if(!f)return;S.file=f;$('#src').value=f.name;
  S.videoURL=URL.createObjectURL(f);vid.src=S.videoURL;vid.hidden=false;$('#vmsg').hidden=true;
  resetData();S.srcName=f.name;setStatus('Đã chọn video');renderKpis();logLine('Đã chọn video: '+f.name)}
$('#uploadBtn').onclick=()=>$('#file').click();
$('#file').onchange=e=>{const fs=[...e.target.files];S.files=fs;CAMRES=[];CAMIDX=-1;renderCamTabs();
  if(fs.length>1){S.file=fs[0];$('#src').value=`${fs.length} video: ${fs.map(f=>f.name).join(', ')}`;S.videoURL=URL.createObjectURL(fs[0]);vid.src=S.videoURL;vid.hidden=false;$('#vmsg').hidden=true;
    resetData();S.srcName=`${fs.length} camera`;setStatus(`Đã chọn ${fs.length} video`);renderKpis();logLine(`Đã chọn ${fs.length} video: mỗi video chạy như một camera, kết quả hiện ở các tab CAM 1, CAM 2...`)}
  else setFile(fs[0]);
  S.cal.per={};renderCalBox()};
$('#src').oninput=()=>{S.file=null;S.files=[];renderCalBox()};
$('#runBtn').onclick=startAnalysis;
$('#stopBtn').onclick=()=>{S.cancel=true;logLine('Đang dừng…')};

/* ---------- Lịch sử phân tích & Xem lại ---------- */
async function loadJobs(){
  const box=$('#jobsList');
  if(!box)return;
  if(!API.ok){
    box.innerHTML='<p class="note">Chưa kết nối Backend. Mở Cài đặt để kết nối máy chủ và xem lịch sử phân tích.</p>';
    return;
  }
  if(API.authRequired&&!API.authenticated){
    box.innerHTML='<p class="note">Máy chủ Backend yêu cầu xác thực. Mở Cài đặt để nhập Mã truy cập Backend (API Key).</p>';
    return;
  }
  try{
    let url='/api/jobs?limit=50';
    if(JOBS_FILTER==='done')url+='&status=done';
    else if(JOBS_FILTER==='active')url+='&status=active';
    else if(JOBS_FILTER==='error')url+='&status=error,cancelled';
    const r=await(await API.f(url)).json();
    const items=r.items||[];
    if(!items.length){
      box.innerHTML='<p class="note empty">Không có lượt phân tích nào.</p>';
      return;
    }
    box.innerHTML=`
      <table class="jobs-table">
        <thead>
          <tr>
            <th>Tên / Nguồn</th>
            <th>Loại</th>
            <th>Trạng thái</th>
            <th>Cảnh báo</th>
            <th>Thời gian</th>
            <th>Thao tác</th>
          </tr>
        </thead>
        <tbody>
          ${items.map(j=>{
            const isSite=j.job_type==='site';
            const typeLabel=isSite?'Đa camera':'1 camera';
            let stCls='done',stText='Hoàn tất';
            if(j.status==='running'){stCls='running';stText='Đang chạy'}
            else if(j.status==='queued'){stCls='queued';stText=`Hàng đợi #${j.queue_position||1}`}
            else if(j.status==='syncing'){stCls='syncing';stText='Đang đồng bộ'}
            else if(j.status==='error'){stCls='error';stText='Lỗi'}
            else if(j.status==='cancelled'){stCls='cancelled';stText='Đã dừng'}
            const tStr=j.created_at?new Date(j.created_at*1000).toLocaleString('vi-VN'):'—';
            const canReplay=j.status==='done';
            const canDelete=j.status!=='running'&&j.status!=='syncing';
            const jobName=j.name||j.job_id;
            return `
              <tr data-job-id="${esc(j.job_id)}">
                <td class="job-name-cell"><b title="${esc(jobName)}">${esc(jobName)}</b></td>
                <td data-label="Loại">${typeLabel}</td>
                <td data-label="Trạng thái"><span class="job-pill ${stCls}">${stText}</span></td>
                <td data-label="Cảnh báo">${j.n_alerts!=null?j.n_alerts:'—'}</td>
                <td data-label="Thời gian"><small>${tStr}</small></td>
                <td class="job-acts-cell">
                  <div class="jobs-acts">
                    <button type="button" class="btn sm" data-job-act="replay" data-job-id="${esc(j.job_id)}" ${canReplay?'':'disabled'}>Xem lại</button>
                    <button type="button" class="btn sm" data-job-act="delete" data-job-id="${esc(j.job_id)}" ${canDelete?'':'disabled'}>Xoá</button>
                  </div>
                </td>
              </tr>
            `;
          }).join('')}
        </tbody>
      </table>
    `;
  }catch(err){
    box.innerHTML=`<p class="note err">Không tải được lịch sử: ${esc(err.message)}</p>`;
  }
}

async function replayJob(jobId){
  if(S.running){
    alert('Đang có tác vụ đang chạy. Hãy đợi hoặc bấm Dừng trước khi xem lại.');
    return;
  }
  logLine(`Đang tải dữ liệu lượt phân tích ${jobId}…`);
  try{
    const job=await API.job(jobId);
    if(job.status!=='done'){
      alert(`Job ${jobId} chưa hoàn tất (${job.status})`);
      return;
    }
    S.jobId=jobId;
    if(job.job_type==='site')await replaySiteJob(job);
    else await replaySingleJob(job);
    window.scrollTo({top:0,behavior:'smooth'});
  }catch(err){
    logLine(`Lỗi tải xem lại: ${err.message}`);
    alert(`Không thể xem lại job: ${err.message}`);
  }
}

async function replaySingleJob(job){
  const id=job.job_id;
  const res=await API.result(id);
  let savedAlerts=[];
  try{
    const alRes=await(await API.f(`/api/alerts?job_id=${id}&limit=500`)).json();
    savedAlerts=alRes.items||[];
  }catch{}
  const alertMap=new Map();
  savedAlerts.forEach(a=>{if(a.event_id!=null)alertMap.set(String(a.event_id),a)});

  const imgs={};
  await Promise.all((res.events||[]).filter(e=>e.task||alertMap.has(String(e.event_id))).map(async e=>{
    const a=alertMap.get(String(e.event_id));
    if(a&&a.img_url)imgs[e.event_id]=await API.mediaUrl(a.img_url);
    else if(e.task)imgs[e.event_id]=await API.blob(id,`files/${job.key||(res.video&&res.video.key)}/vlm_inputs/${e.task}/sheet.jpg`);
  }));

  const videoURL=await API.mediaUrl(`/api/jobs/${id}/video`);
  S.progress=100;
  await pollMap(id,()=>true,true);
  const dataObj=fromResult(res,e=>imgs[e.event_id]);
  await Promise.all((dataObj.events||[]).map(async e=>{
    const a=alertMap.get(String(e.id));
    if(a&&a.review)e.review=a.review;
    if(a&&a.img_url&&!imgs[e.id])e.img=await API.mediaUrl(a.img_url);
  }));

  loadData(dataObj,videoURL,{liveVideo:false,name:job.name||('Xem lại: '+id)});
  setStatus(`Hoàn tất (Xem lại: ${job.name||id})`);
  logLine(`Đã tải kết quả xem lại lượt phân tích ${job.name||id}.`);
}

async function replaySiteJob(job){
  const id=job.job_id;
  const res=await(await API.f(`/api/site-jobs/${id}/result`)).json();
  SITEDATA={
    bounds:res.bounds,cams:res.cameras,frames:res.frames,fps:res.fps,events:res.events||[],
    artifacts:res.artifacts||[],err:res.events_error||null,engine:res.engine||[],
    mapSource:res.map_source||'',mapNote:res.map_note||null
  };
  if(res.map)SITEDATA.mapImg=await loadImg(`/api/site-jobs/${id}/map`);
  SITEDATA.heat=computeSiteHeat();
  SITEDATA.view=computeSiteView();
  SITEDATA.analytics=EFAnalytics.computeMuseumAnalytics({heatmap:SITEDATA.heat,cams:SITEDATA.cams,exhibits:SITEDATA.artifacts});
  const dur=(res.frames&&res.frames.length)?(res.frames.length/(res.fps||1)):60;

  let savedAlerts=[];
  try{
    const alRes=await(await API.f(`/api/alerts?job_id=${id}&limit=500`)).json();
    savedAlerts=alRes.items||[];
  }catch{}

  CAMRES=[];CAMIDX=-1;renderCamTabs();
  const cams=res.cameras||[];
  for(const cm of cams){
    const camId=cm.id||cm;
    const evs=await Promise.all((SITEDATA.events||[]).filter(e=>e.cams&&e.cams.includes(camId)).map(async e=>{
      const a=savedAlerts.find(x=>String(x.event_id)===String(e.id));
      return{
        id:e.id,track:e.person_id!=null?e.person_id:'?',art:e.artifact!=null?e.artifact:'?',
        start:e.t0,end:e.t1,behavior:e.behavior,decision:'ANOMALY',conf:e.conf,
        reason:e.evidence||'Engine phát hiện hành vi bất thường.',review:a?a.review:null,
        img:a&&a.img_url?await API.mediaUrl(a.img_url):null
      };
    }));
    const data={real:true,raw:null,meta:{duration:dur,camera_id:cm.label||`Camera ${camId}`,width:1920,height:1080,nw:960},arts:[],tracks:[],events:evs};
    const camVideoURL=await API.mediaUrl(`/api/site-jobs/${id}/video/${camId}`);
    CAMRES.push({name:`Camera ${camId}`,label:cm.label||camId,cam:camId,data,videoURL:camVideoURL,map:MAP});
  }
  S.progress=100;
  if(CAMRES.length)showCam(0);
  setStatus(`Hoàn tất (${CAMRES.length} camera) · Xem lại: ${job.name||id}`);
  logLine(`Đã khôi phục xem lại địa điểm ghép ${CAMRES.length} camera.`);
}

/* ---------- Điều khiển giao diện ---------- */
function setView(v){['dash','set'].forEach(k=>$('#v-'+k).hidden=k!==v);$$('#tabs .tab').forEach(b=>b.classList.toggle('on',b.dataset.view===v));const vt=$('#viewTitle');if(vt)vt.textContent=v==='set'?'Cài đặt hệ thống':'Giám sát phòng trưng bày';window.scrollTo(0,0);if(v==='dash'){_panelsAt=0;requestAnimationFrame(()=>{paint();if(typeof redrawAnalyticsCharts==='function')redrawAnalyticsCharts()})}}
$('#tabs').onclick=e=>{const b=e.target.closest('[data-view]');if(b)setView(b.dataset.view)};
$('#beChip').onclick=()=>setView('set');
$('#refreshBtn').onclick=()=>API.ping(true);
$('#play').onclick=()=>{if(S.t>=D-.05)seek(0);setPlaying(!S.playing)};
$('#speed').onchange=e=>{S.speed=+e.target.value;vid.playbackRate=S.speed};
$('#layers').onclick=e=>{const b=e.target.closest('[data-l]');if(!b)return;S.layers[b.dataset.l]=!S.layers[b.dataset.l];b.classList.toggle('on')};
$('#tl').onclick=e=>{if(e.target.dataset.seek)return seek(+e.target.dataset.seek);const r=$('#tl').getBoundingClientRect();seek((e.clientX-r.left)/r.width*D)};
$('#tl').onkeydown=e=>{
  if(!S.loaded&&!S.videoURL)return;
  if(e.key==='ArrowLeft'){e.preventDefault();seek(S.t-5)}
  else if(e.key==='ArrowRight'){e.preventDefault();seek(S.t+5)}
  else if(e.key==='Home'){e.preventDefault();seek(0)}
  else if(e.key==='End'){e.preventDefault();seek(D)}
};

$('#filters').onclick=e=>{const b=e.target.closest('[data-f]');if(!b)return;S.filter=b.dataset.f;$$('#filters .tg').forEach(x=>x.classList.toggle('on',x===b));renderAlerts()};
const _ass=$('#alertScopeSel');
if(_ass)_ass.onclick=e=>{const b=e.target.closest('[data-scope]');if(!b)return;S.alertScope=b.dataset.scope;$$('#alertScopeSel .tg').forEach(x=>x.classList.toggle('on',x===b));renderAlerts()};

const _jf=$('#jobsFilters');
if(_jf)_jf.onclick=e=>{const b=e.target.closest('[data-jf]');if(!b)return;JOBS_FILTER=b.dataset.jf;$$('#jobsFilters .tg').forEach(x=>x.classList.toggle('on',x===b));loadJobs()};
const _jrb=$('#jobsRefreshBtn');
if(_jrb)_jrb.onclick=()=>loadJobs();
const _jl=$('#jobsList');
if(_jl)_jl.onclick=async e=>{
  const b=e.target.closest('[data-job-act]');if(!b)return;
  const act=b.dataset.jobAct,jid=b.dataset.jobId;
  if(act==='replay')replayJob(jid);
  else if(act==='delete'){
    if(!confirm('Bạn có chắc chắn muốn xoá lượt phân tích này không?'))return;
    try{await API.f(`/api/jobs/${jid}`,{method:'DELETE'});loadJobs()}catch(err){alert('Lỗi xoá job: '+err.message)}
  }
};

$('#alerts').onclick=async e=>{
  const sk=e.target.closest('[data-seek]');if(sk){seek(+sk.dataset.seek);return}
  const grv=e.target.closest('[data-global-rv]');
  if(grv){
    const aid=grv.dataset.alertId,rv=grv.dataset.globalRv;
    try{
      await API.f(`/api/alerts/${encodeURIComponent(aid)}`,{method:'PATCH',headers:{'Content-Type':'application/json'},body:JSON.stringify({review:rv})});
      const it=GLOBAL_ALERTS.find(x=>x.alert_id===aid);if(it)it.review=rv;
      renderAlerts();
    }catch(err){alert('Lỗi cập nhật đánh giá: '+err.message)}
    return;
  }
  const rv=e.target.closest('[data-rv]');
  if(rv){
    const ev=DATA.events.find(x=>x.id==rv.dataset.id);
    if(ev){
      const reviewVal=rv.dataset.rv;ev.review=reviewVal;renderAlerts();
      if(API.isReady()){
        const alertId=ev.alert_id||(S.jobId?`${S.jobId}:${ev.id}`:null);
        if(alertId){
          try{await API.f(`/api/alerts/${encodeURIComponent(alertId)}`,{method:'PATCH',headers:{'Content-Type':'application/json'},body:JSON.stringify({review:reviewVal})})}catch{}
        }
      }
    }
  }
};

$('#csvBtn').onclick=async()=>{
  if(S.alertScope==='global'){
    if(!API.ok){alert('Vui lòng kết nối Backend trước khi xuất CSV toàn lịch sử.');return}
    if(API.authRequired&&!API.authenticated){alert('Vui lòng xác thực Backend (nhập Mã truy cập API Key) trước khi xuất CSV toàn lịch sử.');return}
    let p='/api/alerts/export.csv?1=1';
    if(S.filter!=='all')p+=`&decision=${encodeURIComponent(S.filter)}`;
    const url=await API.mediaUrl(p);
    const a=document.createElement('a');a.href=url;a.download='exhibitflow_alerts.csv';a.click();
    return;
  }
  const rows=[['id','bat_dau_s','ket_thuc_s','khach','hien_vat','hanh_vi','quyet_dinh','tin_cay','danh_gia','ghi_chu_vlm']];
  const list=S.filter==='all'?DATA.events:DATA.events.filter(e=>e.decision===S.filter);
  list.forEach(e=>{const a=DATA.arts.find(a=>a.id===e.art);
    rows.push([e.id,e.start.toFixed(2),e.end.toFixed(2),e.track,a?a.name:e.art,e.behavior,e.decision,e.conf==null?'':e.conf,e.review||'',e.reason])});
  const csv='﻿'+rows.map(r=>r.map(v=>`"${String(v).replace(/"/g,'""')}"`).join(',')).join('\r\n');
  const a=document.createElement('a');a.href=URL.createObjectURL(new Blob([csv],{type:'text/csv;charset=utf-8'}));a.download='exhibitflow_alerts.csv';a.click()};
document.addEventListener('keydown',e=>{if(e.code==='Space'&&!$('#v-dash').hidden&&!/SELECT|INPUT|BUTTON|TEXTAREA/.test(e.target.tagName)){e.preventDefault();$('#play').click()}});

/* ---------- Settings: kết nối, nhập kết quả ---------- */
function applyDirectUI(){
  $('#beDirect').checked=API.direct;
  $('#beUrl').disabled=API.direct;
  $('#beOff').disabled=API.direct;
  $('#beUrl').value=API.direct?'':(API.base||(()=>{try{return localStorage.getItem('ef_api')||''}catch{return ''}})());
  const beKeyInput=$('#beKey');
  if(beKeyInput)beKeyInput.value=API.key||'';
  const beRemInput=$('#beRememberKey');
  if(beRemInput)beRemInput.checked=API.rememberKey;
}
$('#beDirect').onchange=async()=>{
  API.direct=$('#beDirect').checked;
  try{localStorage.setItem('ef_direct',API.direct?'1':'0')}catch{}
  API.base=API.direct?'':(()=>{try{return localStorage.getItem('ef_api')||''}catch{return ''}})();
  API._pingSeq++;
  API._sessionEpoch++;
  API.ok=false;
  API._gpuOnline=false;
  API._gpuUrl='';
  API._sitesNeedRetry=false;
  disconnectGlobalSSE();
  applyDirectUI();
  await API.ping(true);
};
applyDirectUI();
$('#beBtn').onclick=async()=>{
  if(!API.direct){
    API.base=$('#beUrl').value.trim().replace(/\/+$/,'');
    try{localStorage.setItem('ef_api',API.base)}catch{}
  }
  const keyInput=$('#beKey');
  const remInput=$('#beRememberKey');
  API.key=(keyInput?keyInput.value:'').trim();
  API.rememberKey=remInput?remInput.checked:false;
  try{
    localStorage.setItem('ef_be_remember_key',API.rememberKey?'1':'0');
    if(API.rememberKey){
      localStorage.setItem('ef_be_key',API.key);
      sessionStorage.removeItem('ef_be_key');
    }else{
      sessionStorage.setItem('ef_be_key',API.key);
      localStorage.removeItem('ef_be_key');
    }
  }catch{}
  API._pingSeq++;
  API._sessionEpoch++;
  API.ok=false;
  API._gpuOnline=false;
  API._gpuUrl='';
  API._sitesNeedRetry=false;
  API.streamTicket='';
  API.streamTicketExpiresAt=0;
  sseErrorCount=0;
  sseFallbackPolling=false;
  disconnectGlobalSSE();
  await API.ping(true);
};
$('#beOff').onclick=async()=>{
  API.base='';
  API.key='';
  API._pingSeq++;
  API._sessionEpoch++;
  API.ok=false;
  API._gpuOnline=false;
  API._gpuUrl='';
  API._sitesNeedRetry=false;
  API.streamTicket='';
  API.streamTicketExpiresAt=0;
  sseErrorCount=0;
  sseFallbackPolling=false;
  $('#beUrl').value='';
  if($('#beKey'))$('#beKey').value='';
  try{
    localStorage.removeItem('ef_api');
    localStorage.removeItem('ef_be_key');
    sessionStorage.removeItem('ef_be_key');
  }catch{}
  disconnectGlobalSSE();
  await API.ping(true);
};
const _cbb=$('#connBannerBtn');
if(_cbb)_cbb.onclick=()=>setView('set');
$('#conf').oninput=e=>$('#confV').textContent=(+e.target.value).toFixed(2);

const _gsb=$('#gpuSaveBtn');
if(_gsb)_gsb.onclick=async()=>{
  if(!API.ok){alert('Vui lòng kết nối Backend trước khi cấu hình GPU Worker.');return}
  if(API.authRequired&&!API.authenticated){alert('Vui lòng xác thực Backend (nhập Mã truy cập API Key) trước khi cấu hình GPU Worker.');return}
  _gsb.disabled=true;_gsb.textContent='Đang lưu…';
  const epoch=API._sessionEpoch;
  const url=$('#gpuUrl').value.trim(),token=$('#gpuToken').value.trim(),payload={url};
  if(token)payload.token=token;
  try{
    const res=await(await API.f('/api/config/gpu',{method:'PUT',headers:{'Content-Type':'application/json'},body:JSON.stringify(payload)})).json();
    if(epoch!==API._sessionEpoch||!API.isReady())return;
    $('#gpuUrl').dataset.syncedUrl=res.url||'';
    $('#gpuToken').value='';
    if(res.has_token)$('#gpuToken').placeholder='đã lưu (nhập mới để đổi)';
    await updateChipAndGpuUI(res.gpu,true);
    logLine(res.gpu&&res.gpu.online?'Kết nối GPU Worker thành công.':'Đã lưu cấu hình nhưng GPU Worker đang ngoại tuyến.');
  }catch(err){alert('Lỗi lưu cấu hình GPU: '+err.message)}
  finally{_gsb.disabled=false;_gsb.textContent='Lưu & Kiểm tra'}
};

/* ---------- Camera (Realtime) ---------- */
let CAMS=(()=>{try{const v=JSON.parse(localStorage.getItem('ef_cams')||'null');if(Array.isArray(v)&&v.length)return v}catch{}return[{name:'CAM 1 · Sảnh chính',url:''},{name:'CAM 2 · Phòng gốm',url:''}]})();
const saveCams=()=>{try{localStorage.setItem('ef_cams',JSON.stringify(CAMS))}catch{}};
const camMode=()=>S.mode==='rt';          // Realtime = camera
function camChoice(){const v=$('#camSel').value;if(v==='__url')return{name:'Camera (URL)',url:$('#camUrl').value.trim()};return CAMS[+v.slice(1)]||{name:'Camera',url:''}}
function renderCams(){
  const cur=$('#camSel').value;
  $('#camSel').innerHTML=CAMS.map((c,i)=>`<option value="c${i}">${esc(c.name)}</option>`).join('')+'<option value="__url">Nhập URL khác…</option>';
  if([...$('#camSel').options].some(o=>o.value===cur))$('#camSel').value=cur;
  $('#camList').innerHTML=CAMS.map((c,i)=>`<div class="camrow"><b>${esc(c.name)}</b><small class="${c.url?'has-url':''}">${esc(c.url||'chưa có URL')}</small><button type="button" class="btn sm" data-del="${i}">Xoá</button></div>`).join('')||'<p class="note">Chưa có camera.</p>';
  applySourceUI()}
function applySourceUI(){
  const rt=S.mode==='rt',cam=camMode();
  $('#srcFile').hidden=cam;$('#srcCam').hidden=!cam;$('#camUrl').hidden=!(cam&&$('#camSel').value==='__url');
  $('#runBtn').textContent=rt?'Bắt đầu':'Phân tích';
  $('#vtitle').textContent=cam?`Camera trực tiếp: ${camChoice().name}`:'Phân tích video (CAM 1)'}
$('#camSel').onchange=applySourceUI;
$('#camAdd').onclick=()=>{const name=$('#camName').value.trim(),url=$('#camAddUrl').value.trim();if(!name)return;CAMS.push({name,url});saveCams();$('#camName').value='';$('#camAddUrl').value='';renderCams()};
$('#camList').onclick=e=>{const b=e.target.closest('[data-del]');if(!b)return;CAMS.splice(+b.dataset.del,1);saveCams();renderCams()};

/* ---------- Khởi động ---------- */
resetData();renderSystem();setStatus('Chưa chạy');renderCams();setMode(S.mode);loadJobs();
logLine('Sẵn sàng.'+(CFG.demo?' Chế độ demo: chưa kết nối backend thì kết quả là dữ liệu mẫu.':''));
setInterval(()=>{$('#clock').textContent=new Date().toLocaleString('vi-VN')},1000);$('#clock').textContent=new Date().toLocaleString('vi-VN');
setInterval(pollSystem,3000);
setInterval(async()=>{
  await API.ping();
  if(sseFallbackPolling&&API.isReady()&&typeof loadJobs==='function'){
    loadJobs();
  }
},5000);
API.ping();requestAnimationFrame(tick);

/* ---------- Ngăn kéo menu ---------- */
(()=>{const app=$('.app'),btn=$('#menuBtn'),scrim=$('#scrim');
  const set=o=>{app.classList.toggle('open',o);scrim.hidden=!o;btn.setAttribute('aria-expanded',o);btn.setAttribute('aria-label',o?'Đóng menu':'Mở menu');
    if(o){const t=$('#tabs .tab.on')||$('#tabs .tab');t&&t.focus()}else if(document.activeElement&&document.activeElement.closest('.side'))btn.focus()};
  btn.onclick=()=>set(!app.classList.contains('open'));
  scrim.onclick=()=>set(false);
  document.addEventListener('keydown',e=>{if(e.key==='Escape'&&app.classList.contains('open'))set(false)});
  $('#tabs').addEventListener('click',e=>{if(e.target.closest('[data-view]'))set(false)});
})();
