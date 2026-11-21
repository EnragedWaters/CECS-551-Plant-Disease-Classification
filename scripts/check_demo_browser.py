"""Exercise the processing walkthrough and sample comparison in a real browser.

Start the demo and export a v2 replay first, then run this script. Requires the
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
    parser.add_argument('--replay', type=Path, default=ROOT/'outputs/phase4-demo-v2/replay-final/offline-demo.html')
    parser.add_argument('--output-dir', type=Path, default=ROOT/'outputs/phase4-demo-v2/browser-check-final')
    parser.add_argument('--browser', help='Optional existing Chromium executable')
    args = parser.parse_args()
    recorded = json.loads((args.replay.parent/'replay.json').read_text())
    args.output_dir.mkdir(parents=True, exist_ok=False)
    errors, checks = [], []
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True, **({'executable_path':args.browser} if args.browser else {}))
        page = browser.new_page(viewport={'width':1440,'height':1100},device_scale_factor=1)
        page.on('pageerror',lambda error:errors.append(str(error)))
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
                page.click('#capture')
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
                checks.append({'mode':kind,'example':i+1,'seven_stages_checked':True})
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
            'play_pause_step_reset_checked':True,'failed_requests_recover_without_counts':True,'offline_http_blocked':True,'browser_errors':errors}
    (args.output_dir/'checks.json').write_text(json.dumps(result,indent=2)+'\n')
    print(f'Walkthrough stages, both datasets, batch counts, pause/resume/reset and offline replay passed: {args.output_dir}')


if __name__=='__main__': main()
