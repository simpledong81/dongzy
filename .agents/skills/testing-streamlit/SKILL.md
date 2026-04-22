# Testing Streamlit UI for dongzy (UHPC-NC app)

Reference knowledge for testing `streamlit_app.py` end-to-end. Learned across PR#7 (6-Tab redesign), PR#8 (deprecation), PR#9 (session_state).

## Boot

```bash
cd /home/ubuntu/dongzy
.venv/bin/streamlit run streamlit_app.py --server.port 8501 --server.headless true
```

Background it, then open `http://localhost:8501` in the already-running Chrome on the VM. Do NOT try to `google-chrome localhost:8501` — Chrome is already up; just use the computer tool / existing tab.

Sample CSV for smoke tests: `data/raw/literature_sample.csv` (20 rows, synthetic). Training takes ~3–5s. Results are **not** for the thesis — the sidebar already warns about this in yellow.

## Clicking Streamlit widgets reliably

`computer(action="left_click", devinid=...)` on Streamlit **tab buttons** is flaky during rerun cycles (the click can land while Streamlit is mid-reconciliation and get dropped). Prefer Playwright over CDP at `http://localhost:29229`:

```python
from playwright.sync_api import sync_playwright
with sync_playwright() as p:
    b = p.chromium.connect_over_cdp("http://localhost:29229")
    ctx = b.contexts[0]
    page = next(pg for pg in ctx.pages if "localhost:8501" in pg.url)

    # tabs
    page.get_by_role("tab", name="③ 特征重要性").click()
    # selectbox — open then pick option
    page.locator('div[data-baseweb="select"]').first.click()
    page.get_by_role("option", name="shear_bond_strength").click()
    # wait for rerun
    page.wait_for_timeout(2500)
```

Run the script from the repo's `.venv`: `.venv/bin/python /tmp/xxx.py`. Playwright browsers (chromium) are installed at the project level.

## DOM tips

- Tabs: `page.get_by_role("tab", name="③ 特征重要性")` — the `aria-selected="true"` attribute confirms activation.
- Selectboxes: `div[data-baseweb="select"]` — `inner_text()` returns current value.
- Dataframes (`st.dataframe`): rendered as `<canvas>`, **not** DOM text. You cannot `inner_text()` the values — you must take a screenshot and read visually, or inspect the underlying CSV product in `/tmp/uhpc_streamlit_*/outputs/`.
- Banners: `page.get_by_text("分析完成").count()` for green `st.success`, `page.get_by_text("显示的是上一次训练的产物").count()` for blue `st.info`, `page.get_by_text("SHAP").count()` for the three-branch SHAP warning.

## Tab③ (session_state) core assertion

PR#9 fixed the bug where changing the target selectbox caused all 6 tabs to disappear. The regression test is: after training, switch the selectbox and assert:

- `page.get_by_role("tab").count() == 6` (tabs preserved)
- `page.get_by_text("显示的是上一次训练的产物").count() == 1` (blue info banner)
- `page.get_by_text("分析完成").count() == 0` (no green success = pipeline not rerun)
- pivot values differ from baseline (visual or CSV comparison)

This is the F3 assertion from `/home/ubuntu/dongzy_pr9_test_plan.md`.

## Tempdir rotation (retrain cleanup)

`streamlit_app.py` uses `tempfile.mkdtemp(prefix="uhpc_streamlit_")` + `st.session_state["uhpc_run_dir"]`. After retrain, `_cleanup_previous_run` deletes the old dir. Assert:

```bash
ls -1d /tmp/uhpc_streamlit_* | wc -l   # should be exactly 1
```

Run once before retrain, once after. The suffix should change (e.g. `w3re3gxb` → `bnyb2txt`), count stays 1.

## Green `st.success` vs blue `st.info` — timing caveat

```python
if run_clicked:
    st.success("分析完成...")
else:
    st.info("显示的是上一次训练的产物...")
```

`run_clicked` is True **only** on the single rerun triggered by clicking the button. Any subsequent rerun (selectbox change, programmatic Playwright interaction, widget focus, etc.) evaluates it False and shows the blue banner. So the green banner is transient — capture it synchronously during the click, don't expect to find it via `get_by_text` afterward. Tempdir rotation is the reliable proof that the pipeline actually reran.

## Expected non-issues (don't flag as bugs)

- **Media file GC 404s** in browser console after a second train run — old `st.image` cached files get GCed. Visible in network tab, invisible to the user.
- **`use_container_width` deprecation warning** — not emitted since PR#8 replaced with `width="stretch"`. If a Devin Review bot claims this signature is invalid ("expected int, got str"), it's a false positive: the bot inspected system Python 3.10 with pre-1.49 streamlit. Confirm with `inspect.signature(st.dataframe)` in `.venv` (streamlit 1.56.0 accepts `width: 'Width' = 'stretch'`).
- **Synthetic data warnings** — sidebar yellow banner "切勿使用仓库自带的 literature_sample.csv 的训练结果写论文" is intentional; don't try to suppress it.

## What's hard / may be broken

- **Streamlit tabs widget is stateless** — `st.tabs()` does not accept `key=` and does not trigger a rerun when the user clicks a different tab. Do not assert tab clicks cause reruns; they don't. Only widgets with `key=` (selectbox, multiselect, number_input, button) trigger reruns.
- **`computer(left_click)` on Streamlit tab buttons** is unreliable during rerun cycles. Always fall back to Playwright CDP for tab/selectbox interactions.
- **Canvas-rendered dataframes** block DOM text extraction. To verify numerical values programmatically, read the underlying CSV in `/tmp/uhpc_streamlit_*/outputs/` (e.g., `permutation_importance.csv`).

## Recording

When testing browser UI, always record:

1. Maximize first: `wmctrl -r :ACTIVE: -b add,maximized_vert,maximized_horz`
2. `computer(action="record_start")`
3. Use `record_annotate` with `type="setup"` for navigation, `type="test_start"` for each named test, `type="assertion"` for outcomes
4. Consolidated assertions — group related checks into one (e.g., "6 tabs visible + blue info banner + no green re-success + target=tensile persisted")
5. `record_stop` with title + summary
6. Upload via `upload_attachment` and attach to the PR comment

## Devin Secrets Needed

None for local testing of `streamlit_app.py` with synthetic CSV. Real-data pipeline runs via `literature-ingest` would need:

- WoS / Scopus / CNKI exports or institutional access (manual user-provided files)
- Optional `MLFLOW_TRACKING_URI` if MLflow tracking is wanted
- No API keys required for OpenAlex or Crossref adapters (public APIs, rate-limited by email User-Agent — use `CROSSREF_MAILTO` env var if Crossref rate limits become an issue)
