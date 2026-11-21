/* Real computations finish before the layer walkthrough plays. Playback controls
   reveal captured outputs; they never invent intermediate category predictions.
   The sample comparison requests one pair at a time and counts it exactly once. */
(async function () {
  const $ = id => document.getElementById(id);
  const modes = ['frozen', 'finetune'];
  const datasets = ['plantvillage', 'plantdoc'];
  const names = {frozen:'Frozen', finetune:'Fine-tuned', plantvillage:'PlantVillage', plantdoc:'PlantDoc'};
  const replay = window.RECORDED_DEMO;
  const lessons = window.DemoLessons;
  const stages = [
    ['Original', 'The photograph', 'Same photo → both models.'],
    ['Prepared', 'Fit the input', 'Keep the whole photo; pad instead of stretching.'],
    ['Early', 'Find local patterns', 'Pixels become pattern responses.'],
    ['Middle', 'Combine patterns', 'Smaller maps combine information from larger areas.'],
    ['Late', 'Build a compact description', '576 small maps feed the classifier.'],
    ['Pooling', 'Many positions → one average', 'Each map becomes one number.'],
    ['Scores', 'Choose a category', 'Highest score wins — even when wrong.']
  ];
  let info, pictures = [], selected = {}, focus = 0, stage = 0, trace = null;
  let capturing = false, playing = false, playTimer = null, showAll = true;
  let batchRunning = false, batchBusy = false, batchTimer = null, completed = [];

  const node = (tag, text, className) => {
    const element = document.createElement(tag);
    if (text !== undefined) element.textContent = text;
    if (className) element.className = className;
    return element;
  };
  function status(text, error=false) {
    $('status').textContent = text;
    $('status').classList.toggle('error', error);
  }
  function fail(error) { status(`This step could not finish: ${error.message}. Check the terminal or use the recorded backup.`,true); }
  async function get(path) {
    const response = await fetch(path, {cache:'no-store'});
    if (!response.ok) throw new Error(`request failed (${response.status})`);
    return response.json();
  }
  function verdict(value) {
    return node('span',value.matches_label ? '✓ Matches label' : '✕ Different label',
      `verdict ${value.matches_label ? 'match' : 'mismatch'}`);
  }
  function bar(score) {
    const track = node('div',undefined,'score-track');
    const fill = node('div',undefined,'score-fill');
    fill.style.width = `${Math.max(0,Math.min(100,score*100))}%`;
    track.append(fill);return track;
  }
  function image(source, alt, className) {
    const img = node('img',undefined,className);img.src=source;img.alt=alt;return img;
  }
  function pauseWalk() { playing=false; clearTimeout(playTimer); playTimer=null; walkControls(); }
  function walkControls() {
    $('capture').disabled = capturing;
    $('capture').textContent = replay ? 'Show recorded stages' : (showAll ? 'Run & show all stages' : 'Run & watch');
    $('show-all').disabled = capturing;
    $('show-all').setAttribute('aria-pressed',String(showAll));
    $('play').disabled = capturing || !trace || playing || (!showAll && stage===6);
    $('pause').disabled = !playing;
    $('back').disabled = capturing || showAll || stage===0;
    $('forward').disabled = capturing || showAll || stage===(trace ? 6 : 1);
    $('walk-tab').disabled = capturing;
    $('batch-tab').disabled = capturing;
    document.querySelectorAll('.focus-button,.thumb').forEach(button => button.disabled=capturing);
    [...$('steps').children].forEach((button,i) => {
      button.disabled=capturing || (!trace && i>1);
      button.setAttribute('aria-current',!showAll && i===stage ? 'step':'false');
    });
  }
  function selectPhoto(index) {
    if (capturing) return;
    pauseWalk();focus=index;stage=0;trace=null;
    const dataset=info.examples[index].dataset;
    selected[dataset]=index;
    renderDataset(dataset);
    datasets.forEach(d=>$(`panel-${d}`).classList.toggle('active',d===dataset));
    $('focus-dataset').textContent=`${names[dataset]} walkthrough`;
    $('focus-label').textContent=`Dataset label: ${pictures[index].label}`;
    $('source').textContent=`Selected source: ${pictures[index].source_path}`;
    $('trace-note').textContent='Playback is slowed for explanation. No learning happens here.';
    renderStage();
    status(replay ? 'Recorded examples ready.' : 'Choose a photo, then run both models.');
  }
  function renderDataset(dataset) {
    const panel=$(`panel-${dataset}`), data=pictures[selected[dataset]];
    panel.querySelector('.selected-photo').src=data.original;
    panel.querySelector('.selected-label').textContent=data.label;
    panel.querySelectorAll('.thumb').forEach(button=>button.setAttribute('aria-pressed',String(Number(button.dataset.index)===selected[dataset])));
  }
  function scoreRows(container, scores) {
    scores.forEach(value=>{
      const row=node('div',undefined,'score-row');
      row.append(node('span',value.label),bar(value.score),node('span',`${(value.score*100).toFixed(1)}%`));
      container.append(row);
    });
  }
  // Single-step and all-stage views share exactly the same captured outputs.
  function renderModelStage(content, mode, step) {
      if (step<2) {
        content.append(image(step===0 ? pictures[focus].original : pictures[focus].prepared,
          step===0 ? 'Selected original photograph' : 'Actual resized and padded model input','stage-photo'));
        if(step===1)lessons.preparation(content,pictures[focus]);
      } else if (step<6 && trace) {
        const item=trace.features[mode][step-2];
        if (step<5) {
          content.append(node('p',`4 of ${item.shape[0]} maps · ${item.shape[1]} × ${item.shape[2]} each`,'feature-info'));
          const grid=node('div',undefined,'feature-grid');
          item.maps.forEach((src,i)=>{
            const figure=node('figure');figure.append(image(src,`${names[mode]} ${item.key} feature channel ${item.channels[i]}`),node('figcaption',`Channel ${item.channels[i]}`));grid.append(figure);
          });content.append(grid);
        } else {
          content.append(image(item.maps[0],'576 pooled feature values arranged as a display grid','pooled'));
          // The display layout is explained inside the optional pooling details.
        }
        const legend=node('div',undefined,'legend');
        legend.append(node('span',`−${item.scale_limit.toFixed(2)}`),node('span',undefined,'gradient'),node('span',`+${item.scale_limit.toFixed(2)}`));
        content.append(legend);
        if(step===5)lessons.pooling(content,trace.predictions[mode].learning,trace.features[mode][2].scale_limit);
      } else if (trace) {
        const value=trace.predictions[mode];
        content.append(node('p',value.label,'predicted-label'),verdict(value));
        const scores=[...value.scores].sort((a,b)=>b.score-a.score);
        scoreRows(content,scores.slice(0,5));
        const more=node('details');more.append(node('summary','All 22 scores'));scoreRows(more,scores.slice(5));content.append(more);
        lessons.classifier(content,value);
        if(value.matches_recorded_prediction===false)content.append(node('p','This predicted category differs from the original evaluation.','notice'));
      } else {
        content.append(node('p','Run the models to show this stage.','caption'));
      }
  }
  function renderStage() {
    $('single-stage').hidden=showAll;
    $('single-lesson').replaceChildren();
    $('all-stages').hidden=!showAll;
    $('all-stages').replaceChildren();
    modes.forEach(mode=>$(`trace-${mode}`).querySelector('.stage-content').replaceChildren());
    $('step-counter').textContent=showAll ? 'All 7 stages' : `Step ${stage+1} of 7`;
    if(showAll) {
      stages.forEach((item,step)=>{
        const section=node('section',undefined,'overview-stage');
        section.dataset.stage=step;
        const heading=node('div',undefined,'stage-explanation');
        heading.append(node('h3',`${step+1}. ${item[1]}`),node('p',item[2]));
        lessons.explain(heading,step);
        const grid=node('div',undefined,'model-grid');
        modes.forEach(mode=>{
          const card=node('article',undefined,'model-card');card.dataset.mode=mode;
          card.append(node('h3',names[mode],'model-heading'));
          const content=node('div',undefined,'stage-content');
          renderModelStage(content,mode,step);card.append(content);grid.append(card);
        });
        section.append(heading,grid);$('all-stages').append(section);
      });
    } else {
      $('stage-title').textContent=stages[stage][1];
      $('stage-description').textContent=stages[stage][2];
      lessons.explain($('single-lesson'),stage);
      modes.forEach(mode=>renderModelStage($(`trace-${mode}`).querySelector('.stage-content'),mode,stage));
    }
    walkControls();
  }
  function advance() {
    if (!playing) return;
    if(stage>=6){pauseWalk();return;}
    stage++;renderStage();
    if(stage===6){pauseWalk();return;}
    playTimer=setTimeout(advance,Number($('speed').value));
  }
  function playWalk() {
    if(!trace || capturing || (!showAll && stage===6))return;
    if(showAll){showAll=false;stage=0;}
    clearTimeout(playTimer);playing=true;renderStage();
    playTimer=setTimeout(advance,Number($('speed').value));
  }
  async function capture() {
    if(capturing)return;
    pauseWalk();capturing=true;trace=null;stage=0;renderStage();
    status(replay ? 'Loading recorded layer outputs…' : 'Running both saved models and observing their real layer outputs…');
    try {
      trace=replay ? replay.records[focus] : await get(`/api/trace/${focus}`);
      if(!trace.observed_logits_equal_plain || !trace.model_states_unchanged)throw new Error('observation checks failed');
      $('trace-note').textContent=replay ? 'Recorded outputs · explanatory playback.' : 'Real outputs · explanatory playback · weights unchanged.';
      status(replay ? 'Recorded stages ready.' : 'Done. Scroll through the stages.');
      capturing=false;renderStage();if(!showAll)playWalk();
    } catch(error) {capturing=false;trace=null;renderStage();fail(error);}
  }
  function batchControls() {
    $('walk-tab').disabled=batchBusy;
    $('batch-start').disabled=batchBusy || batchRunning || completed.length===info.batch_steps;
    $('batch-start').textContent=completed.length===info.batch_steps ? 'Comparison complete' : completed.length ? 'Resume comparison' : (replay ? 'Play recorded comparison':'Start comparison');
    $('batch-pause').disabled=!batchRunning;
    $('batch-next').disabled=batchBusy || batchRunning || completed.length===info.batch_steps;
    $('batch-reset').disabled=batchBusy || (completed.length===0 && !batchRunning);
    $('batch-progress').textContent=`${completed.length} / ${info.batch_steps} per dataset`;
    $('progress').value=completed.length;
  }
  function pauseBatch() {
    batchRunning=false;clearTimeout(batchTimer);batchTimer=null;batchControls();
    status(batchBusy ? 'Pausing after the current image pair finishes.' : 'Sample comparison paused. Resume or advance one pair.');
  }
  function totals(dataset,mode) {
    const correct=completed.filter(step=>step.records[dataset].predictions[mode].matches_label).length;
    return {correct,wrong:completed.length-correct,total:completed.length};
  }
  function renderBatch() {
    datasets.forEach(dataset=>{
      const panel=$(`batch-${dataset}`), last=completed.at(-1)?.records[dataset];
      const photo=panel.querySelector('.batch-photo');photo.hidden=!last;
      panel.querySelector('.batch-label').textContent=last ? `Dataset label: ${last.label}` : 'Waiting to start';
      const predictions=panel.querySelector('.batch-predictions');predictions.replaceChildren();
      if(last) {
        photo.src=last.original;
        const source=node('details');source.append(node('summary','Image source'),node('p',last.source_path,'source-path'));predictions.append(source);
        modes.forEach(mode=>{
          const p=node('p');p.append(node('strong',`${names[mode]}: `),node('span',last.predictions[mode].label),node('br'),verdict(last.predictions[mode]));predictions.append(p);
        });
      }
      const output=panel.querySelector('.batch-totals');output.replaceChildren();
      modes.forEach(mode=>{
        const count=totals(dataset,mode),accuracy=count.total ? count.correct/count.total : 0;
        const item=node('div',undefined,'total');item.dataset.mode=mode;
        item.append(node('strong',names[mode]),node('strong',count.total ? `${(accuracy*100).toFixed(1)}%` : '—'),bar(accuracy),node('small',`${count.correct} correct · ${count.wrong} incorrect · ${count.total} processed`));output.append(item);
      });
    });
    $('batch-log').replaceChildren();
    completed.forEach((step,i)=>datasets.forEach(dataset=>{
      const record=step.records[dataset],row=node('tr');
      [i+1,names[dataset],record.label,record.predictions.frozen.label,record.predictions.finetune.label].forEach(text=>row.append(node('td',String(text))));
      $('batch-log').append(row);
    }));
    batchControls();
  }
  async function nextPair() {
    if(batchBusy || completed.length>=info.batch_steps)return;
    batchBusy=true;batchControls();
    const index=completed.length;
    status(`${replay ? 'Reading recorded' : 'Running'} pair ${index+1}: one PlantVillage image and one PlantDoc image…`);
    try {
      const record=replay ? replay.batch[index] : await get(`/api/batch/${index}`);
      if(record.index!==index)throw new Error('unexpected sample order');
      completed.push(record);batchBusy=false;renderBatch();
      if(completed.length===info.batch_steps){batchRunning=false;batchControls();status('Sample complete · 22 images per dataset, not the full evaluation.');}
      else if(batchRunning)batchTimer=setTimeout(nextPair,Number($('batch-speed').value));
      else status('Pair complete. Resume or choose Next pair.');
    } catch(error){batchBusy=false;batchRunning=false;batchControls();fail(error);}
  }
  function switchView(batch) {
    if(capturing)return;
    pauseWalk();if(batchRunning)pauseBatch();
    $('walk-view').hidden=batch;$('batch-view').hidden=!batch;
    $('walk-tab').setAttribute('aria-pressed',String(!batch));$('batch-tab').setAttribute('aria-pressed',String(batch));
    status(batch ? 'Start or step through the sample.' : 'Choose a photo to explore.');
  }

  try {
    info=replay ? replay.info : await get('/api/info');
    if(info.schema_version!==3)throw new Error('old server or backup; restart the launcher or regenerate the backup');
    $('mode').textContent=replay ? 'Recorded replay · offline' : `Live computation · ${info.device.toUpperCase()}`;
    $('mode').classList.toggle('recorded',!!replay);
    $('capture').textContent=replay ? 'Watch recorded processing' : 'Run & watch';
    $('selection').textContent=info.selection_policy;
    $('batch-policy').textContent='One image per category per dataset · selected without using predictions.';
    $('progress').max=info.batch_steps;
    pictures=await Promise.all(info.examples.map((e,i)=>replay ? replay.records[i] : get(`/api/example/${i}`)));
    datasets.forEach(dataset=>{
      const indexes=info.examples.map((e,i)=>e.dataset===dataset ? i : -1).filter(i=>i>=0);
      selected[dataset]=indexes[0];
      const panel=$(`panel-${dataset}`);
      indexes.forEach(i=>{
        const button=node('button',undefined,'thumb');button.dataset.index=i;
        button.setAttribute('aria-label',`${names[dataset]}: ${info.examples[i].purpose}`);
        button.append(image(pictures[i].original,info.examples[i].label),node('span',info.examples[i].purpose));
        button.onclick=()=>selectPhoto(i);panel.querySelector('.thumbnails').append(button);
      });
      panel.querySelector('.focus-button').onclick=()=>selectPhoto(selected[dataset]);renderDataset(dataset);
    });
    stages.forEach((s,i)=>{const button=node('button',s[0]);button.onclick=()=>{pauseWalk();showAll=false;stage=i;renderStage();};$('steps').append(button);});
    $('show-all').onclick=()=>{pauseWalk();showAll=true;renderStage();};
    $('capture').onclick=capture;$('play').onclick=playWalk;$('pause').onclick=pauseWalk;
    $('back').onclick=()=>{pauseWalk();stage=Math.max(0,stage-1);renderStage();};
    $('forward').onclick=()=>{pauseWalk();stage=Math.min(trace ? 6 : 1,stage+1);renderStage();};
    $('speed').onchange=()=>{if(playing)playWalk();};
    $('walk-tab').onclick=()=>switchView(false);$('batch-tab').onclick=()=>switchView(true);
    $('batch-start').onclick=()=>{if(!batchBusy){batchRunning=true;batchControls();nextPair();}};
    $('batch-pause').onclick=pauseBatch;
    $('batch-next').onclick=()=>{pauseBatch();nextPair();};
    $('batch-reset').onclick=()=>{pauseBatch();completed=[];renderBatch();status('Sample reset. Repeating it does not create new test data.');};
    lessons.flow($('dimension-flow'));renderBatch();selectPhoto(0);
  } catch(error){fail(error);}
})();
