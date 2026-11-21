"""Exercise the processing walkthrough and sample comparison in a real browser.

Start the demo and export a learning replay first, then run this script. Requires the
optional Playwright browser tools. All screenshots/PDFs contain dataset images
and are written to a NEW ignored output folder. No training or uploads occur.
"""
import argparse
import json
from pathlib import Path
from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--url', default='http://localhost:8766')
    parser.add_argument('--replay', type=Path, default=ROOT/'outputs/demo-visual-cleanup/replay/offline-demo.html')
    parser.add_argument('--output-dir', type=Path, default=ROOT/'outputs/demo-visual-cleanup/browser-check')
    parser.add_argument('--browser', help='Optional existing Chromium executable')
    args = parser.parse_args()
    recorded = json.loads((args.replay.parent/'replay.json').read_text())
    args.output_dir.mkdir(parents=True, exist_ok=False)
    errors, checks, api_requests = [], [], []
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True, **({'executable_path':args.browser} if args.browser else {}))
        page = browser.new_page(viewport={'width':1440,'height':1100},device_scale_factor=1)
        page.on('pageerror',lambda error:errors.append(str(error)))
        page.on('request', lambda request: api_requests.append(request.url) if '/api/' in request.url else None)
        for kind,url in [('live',args.url),('recorded',args.replay.resolve().as_uri())]:
            # The file:// replay must function even when all HTTP is blocked.
            if kind=='recorded': page.route('http://**/*',lambda route:route.abort())
            page.goto(url)
            page.locator('#capture:enabled').wait_for()
            assert ('Recorded' in page.locator('#mode').inner_text()) == (kind=='recorded')
            assert page.locator('#panel-plantvillage').is_visible()
            assert page.locator('#panel-plantdoc').is_visible()
            assert page.locator('.thumb').count()==6
            if kind=='live':
                # A failed request must recover without inventing a result or count.
                page.route('**/api/trace/*',lambda route:route.fulfill(status=500,body='test failure'))
                page.click('#capture')
                page.locator('#status.error').wait_for()
                assert page.locator('#play').is_disabled()
                assert page.locator('#capture').is_enabled()
                page.unroute('**/api/trace/*')
                page.click('#batch-tab')
                page.route('**/api/batch/*',lambda route:route.fulfill(status=500,body='test failure'))
                page.click('#batch-next')
                page.locator('#status.error').wait_for()
                assert page.locator('#batch-progress').inner_text()=='0 / 22 per dataset'
                assert page.locator('#batch-next').is_enabled()
                page.unroute('**/api/batch/*')
                page.click('#walk-tab')
            for i in range(6):
                page.click(f'.thumb[data-index="{i}"]')
                page.click('#show-all')
                # Selecting a different photo must clear the previous trace.
                assert page.locator('#all-stages .feature-grid img').count()==0
                page.click('#capture')
                page.locator('#all-stages .predicted-label').first.wait_for()
                assert page.locator('#all-stages .overview-stage').count()==7
                assert page.locator('#all-stages .feature-grid img').count()==24
                assert page.locator('#all-stages .pooled').count()==2
                assert page.locator('#all-stages .score-row').count()==44
                page.wait_for_function("Array.from(document.querySelectorAll('#all-stages img')).every(i => i.complete && i.naturalWidth > 0)")
                for mode in ['frozen','finetune']:
                    assert page.locator(f'#all-stages [data-mode="{mode}"] .predicted-label').inner_text()==recorded['records'][i]['predictions'][mode]['label']
                # The default view should stay concise while showing all real stages.
                assert len(page.locator('body').inner_text().split()) < 700
                assert page.locator('#all-stages .stage-purpose[open]').count()==0
                assert page.locator('#all-stages .pool-mini span').count()==98
                assert page.locator('.training-lesson').get_attribute('open') is None
                if i==5:
                    page.screenshot(path=str(args.output_dir/f'{kind}-visual-summary.png'),full_page=True)
                    if kind=='recorded':
                        page.pdf(path=str(args.output_dir/'visual-summary-backup.pdf'),format='A4',print_background=True,scale=.65)
                # Lessons must use this photo's captured values without another inference.
                request_count=len(api_requests)
                assert page.locator('#dimension-flow > span').count()==7
                assert page.locator('#all-stages .stage-purpose').count()==7
                assert page.locator('#all-stages .content-outline').count()==2
                toy=page.locator('#all-stages .filter-lesson')
                toy.locator('summary').click()
                for pattern,response in [('flat',0),('edge',3),('reverse',-3)]:
                    toy.locator('select').select_option(pattern)
                    assert toy.locator('.toy-result').inner_text().endswith(str(response))
                for mode in ['frozen','finetune']:
                    learn=recorded['records'][i]['predictions'][mode]['learning']
                    card=page.locator(f'#all-stages [data-stage="5"] [data-mode="{mode}"]')
                    card.locator('.pool-lesson summary').click()
                    for channel in range(4):
                        card.locator('select').select_option(str(channel))
                        assert card.locator('.pool-grid span').count()==49
                        assert card.locator('.pool-average').inner_text()==f"{learn['pooling'][channel]['observed']:.3f}"
                        assert f"average {learn['pooling'][channel]['observed']:.4f}" in card.locator('.pool-preview').inner_text()
                        assert f"Sum {learn['pooling'][channel]['sum']:.4f}" in card.locator('.pool-equation').inner_text()
                        assert abs(page.locator(f'#all-stages [data-stage="5"] [data-mode="{mode}"] .pooled-marker').evaluate('(n)=>parseFloat(n.style.left)')-channel/24*100)<1e-4
                    classifier=page.locator(f'#all-stages [data-stage="6"] [data-mode="{mode}"] .classifier-lesson')
                    classifier.locator('summary').click()
                    for category in learn['classifier']:
                        classifier.locator('select').select_option(str(category['class_id']))
                        assert classifier.locator('.contribution-row').count()==5
                        assert f"Actual raw score: {category['actual_logit']:.4f}" in classifier.locator('.classifier-total').inner_text()
                    prep=page.locator(f'#all-stages [data-stage="1"] [data-mode="{mode}"] .learning-details')
                    prep.locator('summary').click()
                    assert prep.locator('.math-line').count()==3
                assert len(api_requests)==request_count
                page.locator('.training-lesson summary').click()
                assert 'Right now both models are fixed.' in page.locator('.training-lesson').inner_text()
                page.set_viewport_size({'width':390,'height':844})
                assert page.evaluate('document.documentElement.scrollWidth <= innerWidth')
                page.set_viewport_size({'width':1440,'height':1100})
                page.locator('.training-lesson summary').click()
                if i==5:
                    page.screenshot(path=str(args.output_dir/f'{kind}-all-stages.png'),full_page=True)
                    if kind=='recorded':
                        page.pdf(path=str(args.output_dir/'all-stages-backup.pdf'),format='A4',print_background=True,scale=.65)
                page.set_viewport_size({'width':390,'height':844})
                assert page.evaluate('document.documentElement.scrollWidth <= innerWidth')
                page.set_viewport_size({'width':1440,'height':1100})
                page.click('#play')
                page.locator('#pause:enabled').wait_for()
                page.click('#pause')
                for step in range(7):
                    page.locator('#steps button').nth(step).click()
                    assert page.locator('#step-counter').inner_text()==f'Step {step+1} of 7'
                    page.wait_for_function("Array.from(document.querySelectorAll('.stage-content img')).every(i => i.complete && i.naturalWidth > 0)")
                    if 2<=step<=4:
                        assert page.locator('.feature-grid img').count()==8
                        assert page.locator('.legend').count()==2
                    if step==5: assert page.locator('.pooled').count()==2
                    if step==6:
                        assert page.locator('.score-row').count()==44
                        for mode in ['frozen','finetune']:
                            value=recorded['records'][i]['predictions'][mode]
                            assert page.locator(f'#trace-{mode} .predicted-label').inner_text()==value['label']
                    if i==5 and step in (2,5,6):
                        page.screenshot(path=str(args.output_dir/f'{kind}-stage-{step}.png'),full_page=True)
                        if kind=='recorded':
                            page.pdf(path=str(args.output_dir/f'processing-stage-{step}.pdf'),format='A4',print_background=True,scale=.65)
                checks.append({'mode':kind,'example':i+1,'seven_stages_checked':True,'all_stages_summary_checked':True})
            # Returning to the summary stops playback without losing the captured results.
            page.click('#show-all')
            assert page.locator('#step-counter').inner_text()=='All 7 stages'
            assert page.locator('#all-stages .feature-grid img').count()==24
            # Play advances; Pause freezes; manual stepping works at the ends.
            page.locator('#steps button').nth(0).click()
            assert page.locator('#back').is_disabled()
            page.select_option('#speed','800');page.click('#play')
            page.wait_for_function("document.querySelector('#step-counter').textContent !== 'Step 1 of 7'")
            page.click('#pause');position=page.locator('#step-counter').inner_text()
            page.wait_for_timeout(1000)
            assert page.locator('#step-counter').inner_text()==position
            page.locator('#steps button').nth(6).click()
            assert page.locator('#forward').is_disabled()
            if kind=='recorded':
                page.pdf(path=str(args.output_dir/'walkthrough-backup.pdf'),format='A4',print_background=True,scale=.65)
            page.click('#batch-tab')
            page.click('#batch-next')
            page.wait_for_function("document.querySelector('#batch-progress').textContent === '1 / 22 per dataset'")
            page.click('#batch-start');page.locator('#batch-pause:enabled').wait_for();page.click('#batch-pause')
            page.locator('#batch-next:enabled').wait_for()
            paused=page.locator('#batch-progress').inner_text()
            page.wait_for_timeout(1800)
            assert page.locator('#batch-progress').inner_text()==paused
            before=int(paused.split()[0]);page.click('#batch-next')
            page.wait_for_function(f"document.querySelector('#batch-progress').textContent === '{before+1} / 22 per dataset'")
            page.click('#batch-reset')
            assert page.locator('#batch-progress').inner_text()=='0 / 22 per dataset'
            assert page.locator('#batch-log tr').count()==0
            page.select_option('#batch-speed','0');page.click('#batch-start')
            page.wait_for_function("document.querySelector('#batch-progress').textContent === '22 / 22 per dataset'",timeout=120000)
            assert page.locator('#batch-log tr').count()==44
            for dataset in ['plantvillage','plantdoc']:
                for mode in ['frozen','finetune']:
                    correct=sum(step['records'][dataset]['predictions'][mode]['label']==step['records'][dataset]['label'] for step in recorded['batch'])
                    total=page.locator(f'#batch-{dataset} .total[data-mode="{mode}"]')
                    assert f'{correct} correct · {22-correct} incorrect · 22 processed' in total.inner_text()
                    assert f'{correct/22*100:.1f}%' in total.inner_text()
            assert page.locator('#batch-start').is_disabled()
            assert page.locator('#batch-next').is_disabled()
            page.screenshot(path=str(args.output_dir/f'{kind}-sample-complete.png'),full_page=True)
            if kind=='recorded':
                page.pdf(path=str(args.output_dir/'sample-backup.pdf'),format='A4',print_background=True,scale=.7)
            page.set_viewport_size({'width':390,'height':844})
            assert page.evaluate('document.documentElement.scrollWidth <= innerWidth')
            page.click('#walk-tab')
            assert page.evaluate('document.documentElement.scrollWidth <= innerWidth')
            page.set_viewport_size({'width':1440,'height':1100})
        assert not errors,errors
        browser.close()
    result={'checks':checks,'batch_pairs_per_mode':22,'batch_counter_checks':True,
            'play_pause_step_reset_checked':True,'failed_requests_recover_without_counts':True,'offline_http_blocked':True,'learning_controls_and_arithmetic_checked':True,'lessons_need_no_extra_inference':True,'headless_only':True,'browser_errors':errors}
    (args.output_dir/'checks.json').write_text(json.dumps(result,indent=2)+'\n')
    print(f'Walkthrough stages, both datasets, batch counts, pause/resume/reset and offline replay passed: {args.output_dir}')


if __name__=='__main__': main()
