/* Presentation controls. Recorded replay and live inference are labeled separately.
   Images are served only by the local Python process, or embedded in the backup. */
(async function () {
  const $ = id => document.getElementById(id);
  const replay = window.RECORDED_DEMO;
  let info, index = 0, busy = false;
  const status = text => { $('status').textContent = text; $('status').classList.remove('error'); };
  const fail = error => { status(`Could not complete this step: ${error.message}. Check the terminal or open the offline backup.`); $('status').classList.add('error'); };
  async function get(path) {
    const response = await fetch(path, {cache:'no-store'});
    if (!response.ok) throw new Error(`request failed (${response.status})`);
    return response.json();
  }
  function controls(locked) {
    busy = locked;
    $('previous').disabled = locked || index === 0;
    $('next').disabled = locked || index === info.count - 1;
    $('example').disabled = locked;
    $('run').disabled = locked;
  }
  async function show(next) {
    if (busy || next < 0 || next >= info.count) return;
    index = next;
    controls(true);
    status('Loading the photograph and preparing its model input…');
    for (const mode of ['frozen','finetune']) {
      const card = $(mode);
      card.classList.remove('match','mismatch');
      card.querySelector('.guess').textContent = replay ? 'Recorded output ready' : 'Ready to run';
      card.querySelector('.verdict').textContent = '—';
      card.querySelector('.score').textContent = '';
    }
    try {
      const data = replay ? replay.records[index] : await get(`/api/example/${index}`);
      $('original').src = data.original;
      $('prepared').src = data.prepared;
      $('dataset').textContent = data.dataset === 'plantdoc' ? 'PlantDoc' : 'PlantVillage';
      $('dimensions').textContent = `${data.original_size[0]} × ${data.original_size[1]} original pixels · display copy shown`;
      $('label').textContent = data.label;
      $('source').textContent = `Dataset-relative source: ${data.source_path}`;
      $('counter').textContent = `${index + 1} of ${info.count}`;
      $('example').value = String(index);
      status(replay ? 'Recorded replay — no model is running. Select “Show recorded predictions”.' : 'Ready. Select “Run both models” to compute predictions now.');
      controls(false);
    } catch (error) { fail(error); controls(false); $('run').disabled = true; }
  }
  async function run() {
    if (busy) return;
    controls(true);
    status(replay ? 'Opening recorded model outputs…' : 'Running both saved models on this photograph…');
    try {
      const data = replay ? replay.records[index] : await get(`/api/predict/${index}`);
      for (const mode of ['frozen','finetune']) {
        const value = data.predictions[mode], card = $(mode);
        card.querySelector('.guess').textContent = value.label;
        card.querySelector('.verdict').textContent = value.matches_label ? 'Matches dataset label' : 'Does not match dataset label';
        card.classList.add(value.matches_label ? 'match' : 'mismatch');
        card.querySelector('.score').textContent = `Model score: ${(100 * value.score).toFixed(1)}% · saved epoch ${value.epoch}${value.matches_recorded_prediction ? '' : ' · prediction differs from original evaluation'}`;
      }
      status(replay ? 'Recorded predictions displayed. This is a backup, not live inference.' : 'Live predictions complete. No training, weight updates, or changes to the original results.');
    } catch (error) { fail(error); }
    finally { controls(false); }
  }
  try {
    info = replay ? replay.info : await get('/api/info');
    $('mode').textContent = replay ? 'Recorded replay · offline' : `Live inference · ${info.device.toUpperCase()}`;
    $('mode').classList.toggle('recorded', !!replay);
    $('run').textContent = replay ? 'Show recorded predictions' : 'Run both models';
    $('selection').textContent = info.selection_policy;
    info.examples.forEach((e,i) => {
      const option = document.createElement('option'); option.value = i;
      option.textContent = `${i+1} · ${e.dataset === 'plantdoc' ? 'PlantDoc' : 'PlantVillage'} · ${e.purpose}`;
      $('example').append(option);
    });
    $('previous').onclick = () => show(index - 1);
    $('next').onclick = () => show(index + 1);
    $('example').onchange = () => show(Number($('example').value));
    $('run').onclick = run;
    document.addEventListener('keydown', event => {
      if (event.target.tagName === 'SELECT' || busy) return;
      if (event.key === 'ArrowRight') {event.preventDefault(); show(index+1);}
      if (event.key === 'ArrowLeft') {event.preventDefault(); show(index-1);}
    });
    await show(0);
  } catch (error) { fail(error); }
})();
