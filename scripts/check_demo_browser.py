"""Check the real demo controls and produce local screenshots/PDF backups.

Start python -m plant_disease.demo first. Then run this script from the repo.
Requires optional playwright and its Chromium browser; no training takes place.
Screenshots contain dataset images, so all generated files stay in outputs/.
"""
import argparse
import json
from pathlib import Path
from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--url',default='http://localhost:8766')
    parser.add_argument('--replay',type=Path,default=ROOT/'outputs/phase4-demo/replay/offline-demo.html')
    parser.add_argument('--output-dir',type=Path,default=ROOT/'outputs/phase4-demo/browser-check')
    parser.add_argument('--browser',help='Optional existing Chromium executable')
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True,exist_ok=False)
    checks = []
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True, **({'executable_path':args.browser} if args.browser else {}))
        page = browser.new_page(viewport={'width':1440,'height':1150},device_scale_factor=1)
        errors = []
        page.on('pageerror',lambda error:errors.append(str(error)))
        for kind,url in [('live',args.url),('recorded',args.replay.resolve().as_uri())]:
            page.goto(url)
            page.locator('#run:enabled').wait_for()
            assert ('Recorded' in page.locator('#mode').inner_text()) == (kind=='recorded')
            for i in range(6):
                page.select_option('#example',str(i))
                page.locator('#run:enabled').wait_for()
                page.click('#run')
                page.wait_for_function("document.querySelector('#frozen .guess').textContent !== 'Ready to run' && document.querySelector('#frozen .guess').textContent !== 'Recorded output ready' && !document.querySelector('#run').disabled")
                assert page.locator('.prediction .score').count()==2
                assert page.locator('.prediction .score').first.inner_text().startswith('Model score:')
                assert 'differs from original' not in page.locator('body').inner_text()
                assert page.locator('img').evaluate_all('(images) => images.every(i => i.complete && i.naturalWidth > 0)')
                checks.append({'mode':kind,'example':i+1,'frozen':page.locator('#frozen .guess').inner_text(),'finetune':page.locator('#finetune .guess').inner_text()})
                if i in (0,4,5):
                    page.screenshot(path=str(args.output_dir/f'{kind}-example-{i+1}.png'),full_page=True)
                if kind=='recorded':
                    page.pdf(path=str(args.output_dir/f'backup-example-{i+1}.pdf'),format='A4',print_background=True,scale=.75)
            page.click('#previous');page.locator('#run:enabled').wait_for()
            assert page.locator('#counter').inner_text()=='5 of 6'
            page.click('#next');page.locator('#run:enabled').wait_for()
            assert page.locator('#counter').inner_text()=='6 of 6'
            assert page.locator('#next').is_disabled()
            # Check narrow windows and horizontal overflow as well as desktop.
            page.set_viewport_size({'width':390,'height':844})
            assert page.evaluate('document.documentElement.scrollWidth <= innerWidth')
            page.set_viewport_size({'width':1440,'height':1150})
        assert not errors,errors
        browser.close()
    (args.output_dir/'checks.json').write_text(json.dumps({'checks':checks,'browser_errors':errors},indent=2)+'\n')
    print(f'Checked live and offline controls for all six examples. Backups: {args.output_dir}')


if __name__ == '__main__':
    main()
