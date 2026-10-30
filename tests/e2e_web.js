const { chromium } = require('playwright');
// End-to-end check of the web app through all three roles.
// Start the server on a throwaway database first, for example:
//   AGRIPRICE_DB=data/e2e.db ADMIN_PASSWORD=e2e-admin python -m uvicorn api.main:app --port 8000
// then:  node tests/e2e_web.js [screenshot-folder]
// Needs Node and Playwright (npm i -D playwright; npx playwright install chromium).
const fs = require('fs');
const out = process.argv[2] || 'e2e-screens';
fs.mkdirSync(out, { recursive: true });
const B = process.env.BASE_URL || 'http://localhost:8000';
const ADMIN = process.env.ADMIN_PASSWORD || 'e2e-admin';
(async () => {
  const b = await chromium.launch();
  const errs = [];
  async function session(name, vp = {width: 390, height: 844}) {
    const ctx = await b.newContext({ viewport: vp, deviceScaleFactor: 2, geolocation: {latitude: -0.61, longitude: 30.66}, permissions: ['geolocation'] });
    const p = await ctx.newPage();
    p.on('pageerror', e => errs.push(`${name}: ${e.message}`));
    p.on('console', m => m.type() === 'error' && errs.push(`${name}: ${m.text()}`));
    return p;
  }
  async function signIn(p, role, contact) {
    await p.goto(B + '/');
    if (role === 'admin') { await p.click('label.choice:has-text("Administrator")'); await p.fill('input[type=password]', ADMIN); }
    else {
      if (role === 'provider') await p.click('label.choice:has-text("Service provider")');
      await p.fill('input[autocomplete=username]', contact); await p.check('input[type=checkbox]');
    }
    await p.click('button:has-text("Sign in")'); await p.waitForTimeout(500);
  }
  const shot = async (p, f) => { await p.waitForTimeout(400); await p.screenshot({ path: `${out}/${f}.png`, fullPage: true }); };

  // 1. provider registers
  const pv = await session('provider');
  await signIn(pv, 'provider', 'trucks@example.com');
  await shot(pv, 'e1-provider-register');
  await pv.click('label.choice:has-text("Transport")');
  await pv.fill('form >> text=Business name >> xpath=.. >> input', 'Ankole Trucks');
  await pv.selectOption('select', 'Mbarara');
  await pv.fill('text=Truck load, kg >> xpath=.. >> input', '5000');
  await pv.fill('text=Price, UGX per trip >> xpath=.. >> input', '350000');
  await pv.fill('textarea', 'Covered 5-tonne truck, Mbarara to western markets.');
  await pv.click('button:has-text("Send for approval")');
  await pv.waitForSelector('text=waiting for approval');
  await shot(pv, 'e2-provider-pending');

  // 2. admin approves
  const ad = await session('admin', {width: 1280, height: 860});
  await signIn(ad, 'admin');
  await ad.waitForSelector('text=Ankole Trucks');
  await shot(ad, 'e3-admin');
  await ad.click('button.btn:text-is("Approve")');
  await ad.waitForSelector('text=is now approved');

  // 3. farmer requests transport
  const fm = await session('farmer');
  await signIn(fm, 'farmer', '0772 123 456');
  await fm.click('label.choice:has-text("Millet")');
  await fm.click('button.chip:has-text("500 kg")');
  await fm.click('button:has-text("Use my location")'); await fm.waitForTimeout(1000);
  await fm.click('button:has-text("Get my advice")'); await fm.waitForSelector('.verdict');
  await shot(fm, 'e4-advice');
  await fm.click('a:has-text("Find transport")');
  await fm.waitForSelector('text=Ankole Trucks');
  await shot(fm, 'e5-services');
  await fm.click('button:has-text("Request a quotation")');
  await fm.check('text=Share the truck >> xpath=.. >> input');
  await shot(fm, 'e6-request-form');
  await fm.click('button:has-text("Send request to Ankole Trucks")');
  await fm.waitForSelector('text=Request sent to Ankole Trucks');
  await fm.click('a:has-text("Follow it in My bookings")');
  await fm.waitForSelector('text=Waiting for quotation');
  await shot(fm, 'e7-bookings-requested');

  // 4. provider quotes
  await pv.reload(); await pv.waitForSelector('text=Send quotation');
  await shot(pv, 'e8-provider-request');
  await pv.fill('text=Your price for this job, UGX >> xpath=.. >> input', '300000');
  await pv.click('button:has-text("Send quotation")'); await pv.waitForSelector('text=Quotation sent');

  // 5. farmer accepts
  await fm.reload(); await fm.waitForSelector('text=Accept quotation');
  await shot(fm, 'e9-bookings-quoted');
  await fm.click('button:has-text("Accept quotation")'); await fm.waitForSelector('text=trucks@example.com');

  // 6. provider loads and completes
  await pv.reload(); await pv.waitForSelector('text=+256772123456');
  await pv.fill('text=Condition when loaded >> xpath=.. >> input', 'Dry, 5 bags');
  await pv.click('button:has-text("Mark as collected")'); await pv.waitForSelector('text=Mark job complete');
  await pv.fill('text=Send the farmer an update >> xpath=.. >> input', 'On the road to Fort Portal');
  await pv.click('button:has-text("Send update")'); await pv.waitForSelector('text=Update sent');
  await shot(pv, 'e10-provider-inprogress');
  await pv.click('button:has-text("Mark job complete")'); await pv.waitForSelector('text=marked as complete');

  // 7. farmer rates
  await fm.reload(); await fm.waitForSelector('text=How was Ankole Trucks');
  await fm.click('button.chip:has-text("5 stars")');
  await fm.fill('text=Price you were paid >> xpath=.. >> input', '5400');
  await fm.click('button:has-text("Send rating")'); await fm.waitForSelector('text=You rated this 5 of 5');
  await fm.click('summary:has-text("History")');
  await shot(fm, 'e11-bookings-done');

  await ad.reload(); await ad.waitForTimeout(600); await shot(ad, 'e12-admin-after');
  if (errs.length) { console.error('Browser errors:', errs); process.exitCode = 1; } else console.log('End-to-end flow passed.');
  await b.close();
})().catch(e => { console.error('FAILED', e.message); process.exit(1); });
