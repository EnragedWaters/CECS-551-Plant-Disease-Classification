/* Teaching controls use captured numbers only. They never modify a model/input.
   The small hand-written filter below is explicitly separate from MobileNet. */
window.DemoLessons = (() => {
  const el = (tag, text, cls) => {
    const n=document.createElement(tag); if(text!==undefined)n.textContent=text;
    if(cls)n.className=cls; return n;
  };
  const number = value => Number(value).toFixed(4);
  const why = [
    ['A photograph + a separate dataset label', 'The label lets us check the answer afterwards. It is never an input to the model.'],
    ['Different photo sizes → 3 color channels × 224 × 224', 'A fixed size lets the same network accept every photo. Padding keeps the whole image without stretching its shape.'],
    ['3 × 224 × 224 → 16 × 112 × 112', 'Learned filters turn pixels into responses to local patterns. A channel is one response map, not a stored photograph.'],
    ['16 × 112 × 112 → 40 × 14 × 14 (several blocks later)', 'Blocks combine responses across larger areas. Fewer positions reduce work while more channels describe different patterns.'],
    ['40 × 14 × 14 → 576 × 7 × 7 (final feature block)', 'Many compact maps give the classifier a richer description. A bright cell is a response, not a diagnosis or attention marker.'],
    ['576 × 7 × 7 → 576 averages', 'One average summarizes each map. This removes its exact layout: different arrangements can have the same average.'],
    ['576 averages → 22 raw scores → 22 softmax scores', 'Each category has its own learned weights. The largest final score selects a category; it can still be confidently wrong.']
  ];
  function explain(container, step) {
    const box=el('details',undefined,'stage-purpose');
    box.append(el('summary','Why this step?'));
    box.append(el('p',why[step][0],'dimension-flow'),el('p',why[step][1]));
    container.append(box);
    if(step===2)container.append(el('p','Colors show responses, not disease regions. Scale is shared between models.','visual-note'),filterLesson());
  }
  function flow(container) {
    container.replaceChildren();
    ['Photo','3 × 224 × 224','16 × 112 × 112','40 × 14 × 14','576 × 7 × 7','576 values','22 scores'].forEach((s,i)=>{
      const box=el('span',s);box.append(el('small',['Original','Prepared','Early','Middle','Late','Average','Classify'][i]));
      container.append(box);if(i<6)container.append(el('b','→'));
    });
  }
  function preparation(container, picture) {
    const p=picture.preparation, frame=el('div',undefined,'prepared-frame');
    // The frame overlays the existing, unchanged prepared image; it is not fed back.
    const img=container.querySelector('.stage-photo'); img.replaceWith(frame);frame.append(img);
    const outline=el('div',undefined,'content-outline');
    outline.style.left=`${p.offset[0]/224*100}%`;outline.style.top=`${p.offset[1]/224*100}%`;
    outline.style.width=`${p.resized_size[0]/224*100}%`;outline.style.height=`${p.resized_size[1]/224*100}%`;frame.append(outline);
    container.append(el('p','Outline = photo · outside = padding', 'caption preparation-geometry'));
    const details=el('details',undefined,'learning-details'); details.append(el('summary','Pixel values & resizing'));
    details.append(el('p',`${p.oriented_size.join(' × ')} → ${p.resized_size.join(' × ')} inside 224 × 224.`));
    details.append(el('p','Each color value goes from 0–255 to 0–1, then is centered and scaled with the same ImageNet settings used during training. This changes the numbers the network receives, not the geometry. The displayed photo stays in ordinary colors.'));
    ['Red','Green','Blue'].forEach((channel,i)=>details.append(el('p',`${channel} at pixel (112, 112): (${p.center_rgb[i]} / 255 − ${p.mean[i]}) / ${p.std[i]} = ${number(p.center_normalized[i])}`, 'math-line')));
    container.append(details);
  }
  function filterLesson() {
    const details=el('details',undefined,'filter-lesson learning-details');
    details.append(el('summary','Try a toy filter (separate example)'));
    details.append(el('p','Hand-written teaching filter, not MobileNet. The real models stay unchanged.'));
    const select=el('select'); select.setAttribute('aria-label','Toy patch pattern');
    [['flat','Uniform patch'],['edge','Dark left, bright right'],['reverse','Bright left, dark right']].forEach(([v,t])=>{const o=el('option',t);o.value=v;select.append(o);});
    const content=el('div',undefined,'toy-calculation');details.append(select,content);
    const kernel=[-1,0,1,-1,0,1,-1,0,1];
    function render(){
      const pixels=select.value==='flat' ? Array(9).fill(1) : (select.value==='edge' ? [0,0,1,0,0,1,0,0,1] : [1,0,0,1,0,0,1,0,0]);
      content.replaceChildren();
      const columns=el('div',undefined,'toy-columns');
      [['Pixel patch',pixels],['× Filter',kernel],['= Products',pixels.map((p,i)=>p*kernel[i])]].forEach(([name,values])=>{
        const part=el('div');part.append(el('strong',name));const grid=el('div',undefined,'toy-grid');
        values.forEach(v=>{const cell=el('span',String(v));cell.dataset.sign=String(Math.sign(v));grid.append(cell);});part.append(grid);columns.append(part);
      });
      const sum=pixels.reduce((s,p,i)=>s+p*kernel[i],0);
      content.append(columns,el('p',`Add the nine products → response = ${sum}`,'toy-result math-line'),el('p','Uniform → zero. Reversed edge → opposite sign.'));
    }
    select.onchange=render;render();return details;
  }
  function pooling(container, learning, limit) {
    const existing=container.querySelector('.pooled'), frame=el('div',undefined,'pooled-frame');
    existing.replaceWith(frame);frame.append(existing);
    const marker=el('span',undefined,'pooled-marker');frame.append(marker);
    // A real map -> its average -> the existing vector is visible without arithmetic.
    const visual=el('div',undefined,'pool-visual');
    const mapPart=el('div'), mini=el('div',undefined,'pool-mini'), mapLabel=el('small','Channel 0 · 7 × 7');
    mapPart.append(mini,mapLabel);
    const averagePart=el('div'), average=el('strong',undefined,'pool-average');averagePart.append(average,el('small','Average'));
    const vectorPart=el('div');vectorPart.append(frame,el('small','576 values'));
    visual.append(mapPart,el('b','→'),averagePart,el('b','→'),vectorPart);container.prepend(visual);
    const preview=el('p',undefined,'pool-preview math-line');
    const details=el('details',undefined,'pool-lesson learning-details');
    details.append(el('summary','Inspect the numbers'));
    const select=el('select');select.setAttribute('aria-label','Pooling channel');
    learning.pooling.forEach(p=>{const o=el('option',`Channel ${p.channel}`);o.value=String(p.channel);select.append(o);});
    const grid=el('div',undefined,'pool-grid'), equation=el('p',undefined,'pool-equation math-line');
    details.append(select,preview,grid,equation,el('p','49 equally weighted positions. Map and average displays use separate scales, each shared between models. The 576-value layout is not image coordinates.'));
    const legend=container.querySelector('.legend');if(legend)details.append(el('p','Scale for the 576-value display:'),legend);
    function render(){
      const p=learning.pooling.find(p=>p.channel===Number(select.value));grid.replaceChildren();mini.replaceChildren();
      average.textContent=p.observed.toFixed(3);mapLabel.textContent=`Channel ${p.channel} · 7 × 7`;
      p.grid.flat().forEach(v=>{const cell=el('span',v.toFixed(2));cell.title=String(v);cell.style.background=`rgba(${v<0?'43,98,177':'207,102,33'},${Math.min(1,Math.abs(v)/limit)*.4})`;grid.append(cell);const tile=el('span');tile.style.background=cell.style.background;tile.title=v.toFixed(4);mini.append(tile);});
      marker.style.left=`${p.channel%24/24*100}%`;marker.style.top=`${Math.floor(p.channel/24)/24*100}%`;
      preview.textContent=`Channel ${p.channel}: 49 positions → average ${number(p.observed)} → one feature value.`;
      equation.textContent=`Sum ${number(p.sum)} ÷ 49 = ${number(p.mean)}. Observed model average: ${number(p.observed)}.`;
    }
    select.onchange=render;render();container.append(details);
  }
  function classifier(container, prediction) {
    const details=el('details',undefined,'classifier-lesson learning-details');details.append(el('summary','Inspect score contributions'));
    details.append(el('p','Feature × weight, summed with a bias (offset). These are arithmetic contributions, not a diagnosis explanation.'));
    const select=el('select');select.setAttribute('aria-label','Classifier category');
    prediction.learning.classifier.forEach(c=>{
      const label=prediction.scores.find(s=>s.class_id===c.class_id).label;
      const option=el('option',`${c.roles.join(' / ')}: ${label}`);option.value=String(c.class_id);select.append(option);
    });
    const terms=el('div',undefined,'contribution-terms'),total=el('p',undefined,'classifier-total math-line');details.append(select,el('p','← Lowers score · Raises score →. Bar scale resets per category.','caption'),terms,total);
    details.append(el('p','Softmax compares all 22 raw scores to create the percentage bars. High scores can still be wrong.'));
    function render(){
      const c=prediction.learning.classifier.find(c=>c.class_id===Number(select.value));terms.replaceChildren();
      const max=Math.max(...c.terms.map(t=>Math.abs(t.product)),1e-9);
      c.terms.forEach(t=>{
        const row=el('div',undefined,'contribution-row');row.dataset.channel=String(t.channel);
        row.append(el('p',`Feature ${t.channel}: ${number(t.value)} × weight ${number(t.weight)} = ${number(t.product)}`));
        const bar=el('div',undefined,'contribution-bar'), fill=el('span');fill.style.width=`${Math.abs(t.product)/max*50}%`;
        fill.className=t.product<0?'negative':'positive';bar.append(fill);row.append(bar);terms.append(row);
      });
      terms.append(el('p',`Other ${c.remaining_count} features together: ${number(c.remaining_sum)}. Bias: ${number(c.bias)}.`, 'remaining-contributions'));
      total.textContent=`All contributions + bias = ${number(c.reconstructed_logit)}. Actual raw score: ${number(c.actual_logit)}.`;
    }
    select.onchange=render;render();container.append(details);
  }
  return {explain,flow,preparation,pooling,classifier};
})();
