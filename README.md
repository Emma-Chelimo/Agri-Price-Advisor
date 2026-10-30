# AgriPrice Advisor

Price forecasts, sell-or-store advice and a services platform for smallholder
farmers in Uganda (Group 6, Makerere University).

## Quick start

Needs Python 3.11 or newer (the results were produced with 3.13). From the project root:

```bash
pip install -r requirements.txt
python scripts/run_pipeline.py --demo-providers   # rebuild every result (~15 min), add demo providers
python -m uvicorn api.main:app --port 8000
```

Then open http://localhost:8000. The built front end is already in `frontend/dist`, so
Node is not needed to run the app. The older Streamlit version still works with
`python -m streamlit run app/app.py` while the web app settles in.

`run_pipeline.py --skip-backtest` reuses the saved backtest (about 3 minutes), and
`--notebook` also re-runs the notebook. Optional settings (AI assistant key,
administrator password) go in `.streamlit/secrets.toml`; copy
`.streamlit/secrets.toml.example` to start.

## Layout

```
AGRI-PRICE/
├── wfp_food_prices_uga.csv          raw WFP Uganda market prices
├── AgriPrice_Advisor_EDA_and_Modeling.ipynb
├── src/
│   ├── config.py                    paths, scope, region map, gap rules, splits
│   ├── data.py                      cleaning and monthly panel pipeline
│   ├── forecasters.py               baselines and models, one shared interface
│   ├── evaluation.py                walk-forward backtest and metrics
│   ├── forecasting.py               model registry and the forecast() the app calls
│   ├── decision.py                  sell-or-store advice, confidence, market comparison
│   ├── flow.py                      sign-in check, nearest market, what the farmer can pick
│   ├── assistant.py                 AI explanations and questions, with a number check
│   ├── advice_eval.py               replays past months to test the advice (Objective 8)
│   └── services.py                  providers, bookings, quotes, ratings, shared trips (SQLite)
├── scripts/
│   ├── run_pipeline.py              runs every step below in order, then the tests
│   ├── build_dataset.py             rebuilds the cleaned data in processed/
│   ├── run_backtest.py              scores every model, writes processed/backtest/
│   ├── select_models.py             picks a method per crop, writes model_registry.json
│   ├── evaluate_advice.py           tests the advice on past prices, writes advice_eval/
│   └── seed_demo_providers.py       adds clearly fictional demo providers (or --remove)
├── tests/
│   ├── test_evaluation.py           checks for look-ahead and split leakage
│   ├── test_forecasting.py          model selection, bands and fallbacks
│   ├── test_decision.py             earnings, break-even, confidence, transport
│   ├── test_flow.py                 phone numbers, emails, location, market choice
│   ├── test_assistant.py            prompts, number check, providers (no real API calls)
│   ├── test_services.py             registry, booking lifecycle, 48-hour rule, shared trips
│   ├── test_api.py                  the web API: sign-in, roles, advice JSON, bookings
│   └── e2e_web.js                   browser test of the whole flow (optional, needs Node)
├── api/
│   ├── main.py                      FastAPI server: every endpoint, and serves frontend/dist
│   ├── auth.py                      signed sign-in tokens (farmer, provider, administrator)
│   ├── serialize.py                 turns advice and bookings into JSON
│   └── settings.py                  reads .streamlit/secrets.toml, then environment variables
├── frontend/                        React + TypeScript web app (built with Vite)
│   ├── src/pages/                   SignIn, Sell, Advice, Services, Bookings, Provider, Admin
│   ├── src/components/              TopBar, Verdict, PriceChart, Prose
│   └── dist/                        the built app the server sends to browsers
├── app/
│   ├── app.py                       older Streamlit UI (sign-in, advice)
│   └── services_ui.py               Part 2 screens: services, bookings, provider portal, admin
├── models/                          trained models (.joblib)
├── processed/                       generated data (do not edit by hand)
├── data/agriprice.db                providers and bookings (created on first use; private)
├── _archive/                        old files kept for reference, safe to delete
├── requirements.txt                 pinned library versions
└── .gitignore                       keeps secrets, contacts and archives out of version control
```

## Rebuilding the data

```bash
python scripts/build_dataset.py
```

The notebook, the build script and the app all import `src/`, so the cleaning
rules exist in one place. To change scope (crops, excluded markets, regions,
gap limits, staleness threshold) edit `src/config.py` and rebuild.

## Evaluating forecasts

```bash
python scripts/run_backtest.py            # validation split, all models (~10 min)
python scripts/run_backtest.py --fast     # baselines only (~20 s)
python scripts/run_backtest.py --test     # also score the held-out test split
python -m pytest -q                       # sanity tests (no look-ahead, splits, baselines)
```

The backtest is walk-forward: at each month *t* a model sees only data up to
*t*, forecasts 1 to 6 months ahead, and is scored against the price WFP
recorded. Validation origins run from 2022-01 to 2023-12 (targets before
2024-01 only); test origins start at 2024-01. Use validation for every
modelling decision and report test once, at the end.

Every model is compared with **naive** (repeat the latest price). Skill is
`1 - MAE(model) / MAE(naive)`; above zero means the model is worth using.
Results go to `processed/backtest/`. To add a model, write a function
`forecast(history, horizon) -> array` and add it in `src/forecasters.py`.

## Forecasts in the app

```bash
python scripts/build_dataset.py     # 1. clean data
python scripts/run_backtest.py --test   # 2. score every method
python scripts/select_models.py     # 3. choose a method per crop
streamlit run app/app.py            # 4. the app reads processed/model_registry.json
```

`select_models.py` picks, for each crop, the method with the best average
validation skill over 1 to 6 months. A method replaces naive only if it beats
it by at least `SELECTION_MIN_SKILL` (2%). It also stores an 80% forecast band
for each crop and horizon, taken from that method's validation errors, and
reports how often real test-period prices fell inside it.

The app calls `src.forecasting.forecast(panel, crop, market, horizon)`. It
forecasts from the latest observed price, never more than 6 months ahead, and
falls back to naive if the chosen method cannot handle a market's history.
When Phase 3 adds better models, register them in `src/forecasters.py`, rerun
steps 2 and 3, and the app picks them up with no code changes.

## Using the app

1. **Sign in** as a farmer with a Ugandan mobile number (07XX XXX XXX or +256 7XX XXX XXX)
   or an email address, and tick the consent box. Providers and buyers sign in the same
   way; the administrator uses the password from `.streamlit/secrets.toml`.
2. **Plan a sale.** Pick the crop and quantity, then tap "Use my location" to find the
   nearest market with recent prices for that crop, or choose one from the list. Storage
   and transport costs can be changed under "Storage and transport costs".
3. **Advice.** The decision (sell now or store, and where), how sure it is, the price
   chart with its likely range, your market against the best alternative, each storage
   option with its break-even price, and a plain-language explanation.
4. **Services and bookings.** From the advice, find storage, transport or a buyer, send a
   request, then accept or decline the quotation under My bookings.

Only crops and markets with a price in the last 6 months are offered. Browsers only share
location on `localhost` or HTTPS, so a deployed server needs HTTPS for "Use my location".

### Changing the front end

Needs Node 20 or newer. Run the API on port 8000, then in another terminal:

```bash
cd frontend
npm install
npm run dev      # live-reloading front end on http://localhost:5173
npm run build    # writes frontend/dist, which the Python server then serves
```

The server signs sign-in tokens with `APP_SECRET` if set, otherwise with a key it creates
in `data/app_secret.key`. Set `AGRIPRICE_DB` to use a different services database, for
example a throwaway one when running `tests/e2e_web.js`.

## AI assistant (optional)

The dashboard can write its explanation with a language model and answer the
farmer's questions about the advice. To turn it on, copy
`.streamlit/secrets.toml.example` to `.streamlit/secrets.toml` and add one key:
an Anthropic key (default model `claude-haiku-5-5`), or any OpenAI-style endpoint
such as OpenAI, Groq or a local Ollama. Without a key the app shows the standard
explanation and the question box explains how to enable it.

How it stays honest (`src/assistant.py`):

- The model receives only a JSON summary of figures the system computed (forecast,
  range, costs, options, chances, market comparison) and a list of things the
  system does not know (weather, pests, buyers' contacts, provider quotes, loans).
- It is told to use only those figures, never change the recommendation, and say
  "I don't have that information" for anything else.
- Every reply is checked before it is shown. Each number must match a figure of
  the same unit in the summary (within 1% for rounding), and any market named
  must be in the summary. A failed explanation falls back to the standard one; a
  failed answer is replaced by a polite refusal.

## Services (Part 2 of the user flow)

```bash
python scripts/seed_demo_providers.py          # optional: 114 fictional demo providers
python scripts/seed_demo_providers.py --remove # take them out again
```

Sign-in asks "I am a": farmer, service provider or buyer, or administrator.

- **Farmers** who follow the advice see ranked options from verified providers:
  buyers (and transport to the suggested market) for "sell now", storage and
  drying for "store". Ranking is by estimated cost (earnings for buyers), then
  rating, then distance. Choosing an option sends a booking request.
- **Providers** register (pending until verified), see requests, send quotations
  or decline, record the produce's condition on arrival, post updates and mark
  the job completed.
- **Administrators** (password `ADMIN_PASSWORD` in `.streamlit/secrets.toml`)
  verify or reject registrations and can suspend providers.
- A request with no quotation after **48 hours** expires and the farmer is offered
  the next ranked provider in one click.
- Contact details are shared only after the farmer accepts a quotation, with a
  map link to the provider's market.
- Before a storage booking's planned sale date, My bookings re-checks the advice
  with the latest prices.
- After completion the farmer rates the provider (1 to 5) and can record the price
  received. Ratings feed the ranking.
- **Shared transport (Objective 6):** transport requests that opt in, on the same
  route with the same provider within 3 days, are shown to the provider as one
  shared trip. The provider quotes one price per trip and each farmer's quotation
  is their share by kg.

Demo providers are named "... (demo)", use @example.com contacts and placeholder
prices. Remove them before any real use. `data/agriprice.db` holds contacts: do not
share it or put it in version control.

## Advice logic

`src/decision.py` documents every formula. In short:

- Earnings from storing *k* months = forecast price × kg left after losses − storage fees.
- The chance storing beats selling now comes from the forecast band (price treated
  as log-normal around the forecast). Storing is advised only at 60% or more.
- Confidence is that chance for "store", or one minus the best storage option's
  chance for "sell now". Labels: High ≥ 75%, Medium ≥ 60%, otherwise Low.
- Market comparison uses only markets whose latest price is from the same month as
  the farmer's market. Transport = 3 UGX per kg per km × haversine distance × 1.3.
- Default costs per crop, with sources, are in `config.CROP_COSTS`; the farmer can
  change them in the app.

## Testing the advice on past prices (Objective 8)

```bash
python scripts/evaluate_advice.py      # about 2 minutes
```

Every past month at every market is replayed as if a farmer with 1,000 kg asked
for advice, using only data up to that month. The advice is scored with the
prices that followed and compared with selling at harvest, always storing 3 or
6 months, and the best timing in hindsight. A move to another market is scored
at that market's price one month later. Results, a confidence check, the worst
cases and sensitivity checks go to `processed/advice_eval/`.

WFP prices are retail prices, so gains from selling at another market are an
upper bound on what a farmer selling in bulk would receive.

## Processed files

| File | Contents |
|---|---|
| `cleaned_prices.parquet` | One row per crop, market, price type and month. Retail and wholesale. |
| `monthly_prices.parquet` | Retail series on a complete monthly calendar (see columns below). |
| `series_summary.csv` | Coverage, gaps, segments, outlier flags and staleness per series. |
| `latest_prices.csv` | Latest observed price per series with its age and a stale flag. |
| `markets.csv` | Market coordinates and regions, for distance calculations. |
| `backtest/` | Walk-forward forecasts, metrics per series, summaries (`run_backtest.py`). |
| `model_registry.json` | Method and forecast bands per crop, read by the app (`select_models.py`). |
| `advice_eval/` | The advice tested on past prices (`evaluate_advice.py`). |

Key columns in `monthly_prices.parquet`:

- `price`: observed price, interpolated for gaps of 2 months or less, missing inside longer gaps
- `price_observed`: the raw WFP value, missing wherever nothing was recorded
- `is_imputed`, `in_long_gap`: how each month was handled
- `segment_id`: increases after every long gap; build lag features within a segment
- `is_outlier_flag`: one-month spike that reverses straight away (flagged, not removed)

## Data rules (Phase 1)

- Six crops: maize, maize flour, beans, sorghum, millet, cassava flour
- 30 markets; the 13 refugee settlement markets are excluded
- Four UBOS regions; Soroti is Eastern, Kiryandongo is Western
- "Maize (white)" is the main maize series; plain "Maize" only fills its missing months
- Dates are stored as the first day of each month
- Staleness is measured against the newest month in the data, not today's date
