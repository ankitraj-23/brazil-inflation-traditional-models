# Data Audit — Raw CEIC CSV Files

**Project:** Phase 2 — traditional econometric models for Brazilian inflation forecasting
(replication of *"Machine learning methods for inflation forecasting in Brazil: new
contenders versus classical models"*).

**Scope of this document:** inspection and standardization of the raw CEIC exports only.
No forecasting model is implemented in this phase.

**Audit date:** 2026-07-06
**Source directory:** `data/raw/` (read-only; not committed — see `.gitignore`)

---

## 1. File structure (common to all six files)

Every raw CEIC CSV has the same two-part layout:

1. **Metadata block** — labelled rows: `Region`, `Subnational`, `Frequency`, `Unit`,
   `Source`, `Status`, `Series ID`, `SR Code`, `Mnemonic`, `Function Description`,
   `First Obs. Date`, `Last Obs. Date`, `Last Update Time`, plus summary statistics
   (`Mean`, `Variance`, `Min`, `Max`, `Median`, `No. of Obs`, …).
2. **Data block** — `MM/YYYY,value` rows (some files have a second, duplicated value
   column).

The metadata block length **varies** between files because the multi-line
`Series Remarks` field pushes the data block down (data starts at line 30 in four files,
line 32 in the two with long remarks). The parser therefore locates the data block by
regex (`^(0[1-9]|1[0-2])/\d{4}$` in column 1), **not** by a fixed row offset.

### Key cross-cutting finding — metadata dates ≠ exported data range

The metadata `First Obs. Date` / `Last Obs. Date` describe the **full CEIC series**
(e.g. IPCA general starts 12/1979, ends 05/2026). However, the **exported data block in
every file already covers exactly 2004-01 … 2018-12 (180 monthly observations)**. The
summary statistics (`Mean`, `Min`, `Max`, `No. of Obs = 180`) are computed over this
2004–2018 window, confirming the files were pre-trimmed to the analysis sample before
export. **Do not read the metadata First/Last Obs dates as the data coverage.**

---

## 2. Audit table

| Raw filename | Clean variable name | Series title (raw) | Region | Frequency | Unit (raw) | Source | First date | Last date | N obs | Covers 2004-01→2018-12 | Concerns / warnings |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `ipca_general_index.csv` | `ipca_general_index` | IPCA: General | Brazil | Monthly | Dec1993=100 | Brazilian Institute of Geography and Statistics (IBGE) | 2004-01 | 2018-12 | 180 | ✅ Yes | Index level (base Dec/1993=100), not a rate. Headline MoM % is derived. Mnemonic `BR.CPI.FICL.M`, Series ID `BRIFAAAAA`. Remarks note weighting-structure changes over time (POF revisions) and a `[COVID-19-IMPACT]` tag — irrelevant to the 2004–2018 window. |
| `ipca_non_regulated_mom.csv` | `ipca_non_regulated_mom` | IPCA: MoM: Non Regulated | Brazil | Monthly | % | Central Bank of Brazil (BCB) | 2004-01 | 2018-12 | 180 | ✅ Yes | Already a MoM % rate. National "free / market" prices half of the IPCA free-vs-administered split. Series ID `BRIFAADIXAAEOK`. |
| `ipca_administered_prices_sp_total_mom.csv` | `ipca_administered_mom` | IPCA: MoM: Administred Prices (SP): Total | **Brazil** | Monthly | % | Central Bank of Brazil (BCB) | 2004-01 | 2018-12 | 180 | ✅ Yes | ⚠️ **"(SP)" ambiguity — see §3.2.** Treated as **national** administered/monitored prices based on metadata (Region=Brazil, empty Subnational) and its pairing with the Non Regulated series. Series ID `BRIFAADIXAAENP`. |
| `broad_money_m4.csv` | `broad_money_m4` | Broad Money Supply: M4 | Brazil | Monthly | BRL mn | Central Bank of Brazil (BCB) | 2004-01 | 2018-12 | 180 | ✅ Yes | ⚠️ **Two value columns, verified identical** over 2004–2018 (spliced vs splice-base of the same M4 series); parser uses the first. Confirmed **M4**, not M1/M2/M3/monetary base — see §3.1. Series ID `408100967 (BRKZCAAAATAANB)`. |
| `exchange_rate_brl_per_usd_period_avg.csv` | `exchange_rate_brl_per_usd` | Exchange Rate against USD: Period Avg: Monthly: Brazil | Brazil | Monthly | `USD/BRL` *(label misleading)* | CEIC Data (history extended from IMF / Federal Reserve) | 2004-01 | 2018-12 | 180 | ✅ Yes | ⚠️ **Unit label says `USD/BRL` but values are BRL per USD** (~2.85–4.11 = reais per dollar). The values are correct for the intended `exchange_rate_brl_per_usd` variable; only the unit string is inverted. See §3.3. Period-average (not end-of-period). |
| `ibc_br_seasonally_adjusted.csv` | `ibc_br_sa` | Economic Activity Index - IBC-Br: Seasonally Adjusted (sa) | Brazil | Monthly | 2022=100 | Central Bank of Brazil (BCB) | 2004-01 | 2018-12 | 180 | ✅ Yes | Central Bank monthly activity proxy, seasonally adjusted, index base 2022=100. Series ID `544340277`. |

**All six files:** 180 observations, no internal date gaps, **zero missing values** in the
2004-01 … 2018-12 window.

---

## 3. Focused verifications

### 3.1 `broad_money_m4.csv` — is it really M4?

**Yes.** Confirmed from metadata, not the filename alone:

- Title (both columns): **`Broad Money Supply: M4`**.
- `Function Description` explicitly references *"Broad Money Supply: M4"* via a CEIC SPLICE
  operation.
- Source: Central Bank of Brazil; Unit: BRL mn; Series ID `408100967 (BRKZCAAAATAANB)`.
- Magnitudes (~0.97–6.76 **trillion** BRL over 2004–2018) are consistent with M4, the
  broadest aggregate — far larger than the monetary base / M1 / M2 / M3.

This is **not** M1, M2, M3, or the monetary base.

**Dual-column note:** the file carries two value columns. They are the spliced series and
its splice base; they are **byte-for-byte identical across the entire 2004–2018 window**
(verified programmatically). The parser uses the first column only.

### 3.2 `ipca_administered_prices_sp_total_mom.csv` — does "SP" mean São Paulo?

**Resolution: treated as NATIONAL administered prices → column `ipca_administered_mom`.**
⚠️ **This is a documented judgement, not a certainty. Read the reasoning before relying on it.**

Evidence that it is **national** (Brazil-wide), not São Paulo-only:

1. **`Region` = Brazil**, and **`Subnational` is empty**. A São Paulo-only series would
   populate `Subnational` (e.g. "São Paulo"). The metadata actively indicates national
   coverage.
2. It **pairs with `ipca_non_regulated_mom`** (Non Regulated + Administered = the standard
   Brazilian IPCA "livres vs. monitorados/administrados" decomposition). Both are BCB MoM
   series with near-adjacent Series IDs (`...AAENP` vs `...AAEOK`).
3. Both are sourced from the **Central Bank of Brazil**, which publishes this
   free/administered breakdown at the **national** level.

Because of this, the column is named `ipca_administered_mom` (national) rather than the
fallback `ipca_administered_candidate_mom`.

**Residual warning:** the raw title literally reads `Administred Prices (SP): Total`. The
`(SP)` token is **unexplained** by the metadata. The most likely reading is that it is a
CEIC series-label/source code, **not** the São Paulo federal unit — but this could not be
confirmed from an authoritative CEIC dictionary within `data/raw/`. If a downstream check
(e.g. reconciling non-regulated + administered against headline IPCA weights) contradicts
the national interpretation, revisit this column and consider renaming it to
`ipca_administered_candidate_mom`.

### 3.3 `exchange_rate_brl_per_usd_period_avg.csv` — direction of the rate

The `Unit` field says `USD/BRL`, but the observed values (~2.85 in 2004, ~3.88 in 2018)
are **BRL per USD** (reais required to buy one US dollar). This matches the intended
variable name and the paper's imported-inflation channel (a weaker real ⇒ higher import
prices). The unit **string** is inverted/mislabelled; the **values** are correct as
`exchange_rate_brl_per_usd`. It is a **period-average** monthly rate, not end-of-period.

---

## 4. Output dataset

`src/prepare_data.py` produces `data/processed/traditional_models_clean_data.csv`:

- **Shape:** 180 rows × 11 columns.
- **`date`:** month-end datetime (`YYYY-MM-DD`), 2004-01-31 … 2018-12-31.
- **Missing values:** exactly **one** NaN each in `ipca_headline_mom`,
  `exchange_rate_mom_pct`, `imported_inflation`, and `m4_diff` — all in the **first row
  (2004-01)**, because the MoM lag / first difference has no prior observation inside the
  exported window (the raw files begin at 2004-01, so no pre-sample value exists). This is
  expected and unavoidable, not a data defect.

---

## 5. Overall assessment

- All six raw series are internally clean: correct frequency (monthly), complete and
  gap-free coverage of the full 2004-01 … 2018-12 sample, and no missing values.
- Two labelling quirks are documented and handled (exchange-rate unit direction; M4
  duplicate columns).
- One identity judgement is documented and flagged (`(SP)` administered prices treated as
  national).
- **The data is ready for the traditional-model phase**, subject to the single
  documented `(SP)` caveat above.
