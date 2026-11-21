/* Real computations finish before the layer walkthrough plays. Playback controls
   reveal captured outputs; they never invent intermediate category predictions.
   The sample comparison requests one pair at a time and counts it exactly once. */
(async function () {
  const $ = id => document.getElementById(id);
  const modes = ['frozen', 'finetune'];
  const datasets = ['plantvillage', 'plantdoc'];
  const names = {frozen:'Frozen', finetune:'Fine-tuned', plantvillage:'PlantVillage', plantdoc:'PlantDoc'};
  const replay = window.RECORDED_DEMO;
  const stages = [
    ['Original', 'Start with the photograph', 'Both saved models receive the same selected photograph. The dataset label is used only to check the answer; it is not given to the model.'],
    ['Prepared', 'Make the image fit the model', 'Correct orientation, convert to RGB, resize without stretching, and pad to 224 × 224. Pixel values are then normalized using the original training settings.'],
    ['Early', 'Early layer responses', 'These are real feature maps: grids of numbers produced inside the network. We show fixed channels 0–3, not handpicked “best-looking” maps.'],
    ['Middle', 'Middle layer responses', 'Later blocks combine earlier responses. Each map is smaller and carries a different learned pattern; we cannot simply name a channel “leaf” or “disease”.'],
    ['Late', 'Late layer responses', 'The final feature block produces 576 small maps. Only the first four are displayed. Their colors are feature responses, not attention or proof of why the model chose a label.'],
    ['Summary', 'Average each feature map into one number', 'Average pooling converts 576 maps into 576 numbers. The grid below arranges those numbers for display; its cells are not locations in the original photograph.'],
    ['Scores', 'Turn those features into category scores', 'The final linear layer scores 22 categories. Softmax converts those scores into values that sum to one. The largest gives the prediction; a high value can still be wrong.']
  ];
  let info, pictures = [], selected = {}, focus = 0, stage = 0, trace = null;
  let capturing = false, playing = false, playTimer = null;
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
    return node('span',value.matches_label ? 'Matches dataset label' : 'Does not match dataset label',
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
    $('play').disabled = capturing || !trace || playing || stage===6;
    $('pause').disabled = !playing;
    $('back').disabled = capturing || stage===0;
    $('forward').disabled = capturing || stage===(trace ? 6 : 1);
    $('walk-tab').disabled = capturing;
    $('batch-tab').disabled = capturing;
    document.querySelectorAll('.focus-button,.thumb').forEach(button => button.disabled=capturing);
    [...$('steps').children].forEach((button,i) => {
      button.disabled=capturing || (!trace && i>1);
      button.setAttribute('aria-current',i===stage ? 'step':'false');
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
    $('trace-note').textContent='Run the models to capture actual layer outputs. Playback is slowed for explanation; it is not inference timing.';
    renderStage();
    status(replay ? 'Recorded backup selected. No models run in this mode.' : 'Choose a dataset photograph, then run and watch its processing.');
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
  function renderStage() {
    $('stage-title').textContent=stages[stage][1];
    $('stage-description').textContent=stages[stage][2];
    $('step-counter').textContent=`Step ${stage+1} of 7`;
    modes.forEach(mode=>{
      const content=$(`trace-${mode}`).querySelector('.stage-content');content.replaceChildren();
      if (stage<2) {
        content.append(image(stage===0 ? pictures[focus].original : pictures[focus].prepared,
          stage===0 ? 'Selected original photograph' : 'Actual resized and padded model input','stage-photo'));
        content.append(node('p',stage===0 ? 'The same photo enters both models.' : 'Actual preprocessing output, before numerical normalization.','caption'));
      } else if (stage<6 && trace) {
        const item=trace.features[mode][stage-2];
        if (stage<5) {
          content.append(node('p',`${item.shape[0]} channels × ${item.shape[1]} × ${item.shape[2]} positions; showing channels 0–3.`,'feature-info'));
          const grid=node('div',undefined,'feature-grid');
          item.maps.forEach((src,i)=>{
            const figure=node('figure');figure.append(image(src,`${names[mode]} ${item.key} feature channel ${item.channels[i]}`),node('figcaption',`Channel ${item.channels[i]}`));grid.append(figure);
          });content.append(grid);
        } else {
          content.append(image(item.maps[0],'576 pooled feature values arranged as a display grid','pooled'));
          content.append(node('p',item.layout_note,'caption'));
        }
        const legend=node('div',undefined,'legend');
        legend.append(node('span',`−${item.scale_limit.toFixed(2)}`),node('span',undefined,'gradient'),node('span',`+${item.scale_limit.toFixed(2)}`));
        content.append(legend,node('p','Shared scale across both models at this stage; pale = zero.','caption'));
      } else if (trace) {
        const value=trace.predictions[mode];
        content.append(node('p',value.label,'predicted-label'),verdict(value));
        const scores=[...value.scores].sort((a,b)=>b.score-a.score);
        scoreRows(content,scores.slice(0,5));
        const more=node('details');more.append(node('summary','Show the other 17 category scores'));scoreRows(more,scores.slice(5));content.append(more);
        content.append(node('p','Final model scores, not calibrated probabilities or a diagnosis.','score-caption'));
        if(value.matches_recorded_prediction===false)content.append(node('p','This predicted category differs from the original evaluation.','notice'));
      }
    });
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
    if(!trace || capturing || stage===6)return;
    clearTimeout(playTimer);playing=true;walkControls();
    playTimer=setTimeout(advance,Number($('speed').value));
  }
  async function capture() {
    if(capturing)return;
    pauseWalk();capturing=true;trace=null;stage=0;renderStage();
    status(replay ? 'Loading recorded layer outputs…' : 'Running both saved models and observing their real layer outputs…');
    try {
      trace=replay ? replay.records[focus] : await get(`/api/trace/${focus}`);
      if(!trace.observed_logits_equal_plain || !trace.model_states_unchanged)throw new Error('observation checks failed');
      $('trace-note').textContent=replay ? 'Recorded model outputs · explanatory playback · no live inference in this backup.' : 'Real outputs captured now. Observed and ordinary predictions match; weights are unchanged. Playback speed is for explanation.';
      status(replay ? 'Recorded outputs ready. Pause or step through them.' : 'Live computation finished. Explore the captured outputs using playback or step controls.');
      capturing=false;renderStage();playWalk();
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
        predictions.append(node('p',`Source: ${last.source_path}`,'caption source-path'));
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
      if(completed.length===info.batch_steps){batchRunning=false;batchControls();status('Sample complete. These counts are for 22 images per dataset and one seed—not the published full evaluation.');}
      else if(batchRunning)batchTimer=setTimeout(nextPair,Number($('batch-speed').value));
      else status('Pair complete. Comparison paused; use Resume or Next pair.');
    } catch(error){batchBusy=false;batchRunning=false;batchControls();fail(error);}
  }
  function switchView(batch) {
    if(capturing)return;
    pauseWalk();if(batchRunning)pauseBatch();
    $('walk-view').hidden=batch;$('batch-view').hidden=!batch;
    $('walk-tab').setAttribute('aria-pressed',String(!batch));$('batch-tab').setAttribute('aria-pressed',String(batch));
    status(batch ? 'This sample uses the same two models on both datasets. Start it or advance one pair at a time.' : 'Select a photograph in either dataset panel to explore its processing.');
  }

  try {
    info=replay ? replay.info : await get('/api/info');
    if(info.schema_version!==2)throw new Error('old server or backup; restart the launcher or regenerate the backup');
    $('mode').textContent=replay ? 'Recorded replay · offline' : `Live computation · ${info.device.toUpperCase()}`;
    $('mode').classList.toggle('recorded',!!replay);
    $('capture').textContent=replay ? 'Watch recorded processing' : 'Run & watch';
    $('selection').textContent=info.selection_policy;
    $('batch-policy').textContent=info.batch_policy;
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
    stages.forEach((s,i)=>{const button=node('button',s[0]);button.onclick=()=>{pauseWalk();stage=i;renderStage();};$('steps').append(button);});
    $('capture').onclick=capture;$('play').onclick=playWalk;$('pause').onclick=pauseWalk;
    $('back').onclick=()=>{pauseWalk();stage=Math.max(0,stage-1);renderStage();};
    $('forward').onclick=()=>{pauseWalk();stage=Math.min(trace ? 6 : 1,stage+1);renderStage();};
    $('speed').onchange=()=>{if(playing)playWalk();};
    $('walk-tab').onclick=()=>switchView(false);$('batch-tab').onclick=()=>switchView(true);
    $('batch-start').onclick=()=>{if(!batchBusy){batchRunning=true;batchControls();nextPair();}};
    $('batch-pause').onclick=pauseBatch;
    $('batch-next').onclick=()=>{pauseBatch();nextPair();};
    $('batch-reset').onclick=()=>{pauseBatch();completed=[];renderBatch();status('Sample reset. Repeating it does not create new test data.');};
    renderBatch();selectPhoto(0);
  } catch(error){fail(error);}
})();
