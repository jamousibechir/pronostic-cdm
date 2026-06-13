"""
Génère un dashboard HTML autonome : outputs/dashboard.html

Vue d'ensemble des pronostics CdM 2026, régénérée à chaque mise à jour :
  - Bilan : matchs joués, taux de bons vainqueurs, score de Brier, calibration
  - Favoris au titre (Monte-Carlo)
  - Tous les matchs : pronostic 1/N/2 + score indicatif vs SCORE RÉEL, badge correct/raté
  - Top Soulier d'or

Lecture : outputs/{matchs,champion,buteurs}.csv + data/fixtures.csv (scores réels).
Aucune dépendance externe : le HTML est autonome (CSS/JS inline), ouvrable hors ligne.

Lancer : python dashboard.py   (puis ouvrir outputs/dashboard.html)
"""
import html
import datetime as dt
import pandas as pd

from config import OUTPUTS_DIR, DATA_DIR

OUT = OUTPUTS_DIR / "dashboard.html"


def _load():
    matchs = pd.read_csv(OUTPUTS_DIR / "matchs.csv") if (OUTPUTS_DIR / "matchs.csv").exists() else pd.DataFrame()
    champ  = pd.read_csv(OUTPUTS_DIR / "champion.csv") if (OUTPUTS_DIR / "champion.csv").exists() else pd.DataFrame()
    boot   = pd.read_csv(OUTPUTS_DIR / "buteurs.csv") if (OUTPUTS_DIR / "buteurs.csv").exists() else pd.DataFrame()
    fixtures = pd.read_csv(DATA_DIR / "fixtures.csv") if (DATA_DIR / "fixtures.csv").exists() else pd.DataFrame()
    return matchs, champ, boot, fixtures


def _merge_actuals(matchs: pd.DataFrame, fixtures: pd.DataFrame) -> pd.DataFrame:
    """Ajoute les scores réels (depuis fixtures) aux pronostics de matchs.csv."""
    if matchs.empty:
        return matchs
    m = matchs.copy()
    m["home_real"] = pd.NA
    m["away_real"] = pd.NA
    if not fixtures.empty:
        fx = fixtures.dropna(subset=["home_score", "away_score"])
        key = {(r["home_team"], r["away_team"]): (r["home_score"], r["away_score"])
               for _, r in fx.iterrows()}
        for i, r in m.iterrows():
            k = (r["home_team"], r["away_team"])
            if k in key:
                m.at[i, "home_real"] = int(key[k][0])
                m.at[i, "away_real"] = int(key[k][1])
    return m


def _outcome(h, a):
    return "1" if h > a else ("N" if h == a else "2")


def _compute_bilan(m: pd.DataFrame):
    """Taux de bons vainqueurs + Brier sur les matchs joués."""
    played = m[m["home_real"].notna()].copy() if not m.empty else pd.DataFrame()
    if played.empty:
        return {"n": 0, "acc": None, "brier": None, "exact": None}
    correct = exact = 0
    brier_sum = 0.0
    graded = 0
    for _, r in played.iterrows():
        if pd.isna(r["p_home_win"]) or pd.isna(r["p_draw"]) or pd.isna(r["p_away_win"]):
            continue   # pronostic non disponible pour ce match -> non noté
        graded += 1
        probs = {"1": float(r["p_home_win"]), "N": float(r["p_draw"]), "2": float(r["p_away_win"])}
        pred = max(probs, key=probs.get)
        real = _outcome(r["home_real"], r["away_real"])
        correct += (pred == real)
        # score exact
        ph, pa = (int(x) for x in str(r["pred_score"]).split("-"))
        exact += (ph == r["home_real"] and pa == r["away_real"])
        oneh = {"1": 0, "N": 0, "2": 0}; oneh[real] = 1
        brier_sum += sum((probs[k] - oneh[k]) ** 2 for k in probs)
    if graded == 0:
        return {"n": len(played), "acc": None, "brier": None, "exact": None,
                "played": played}
    return {"n": graded, "acc": correct / graded, "brier": brier_sum / graded,
            "exact": exact / graded, "played": played}


# ─────────────────────────────────────────────────────────────────────────────
# Rendu HTML
# ─────────────────────────────────────────────────────────────────────────────

def _bar_row(label, pct, sub="", cls="bar"):
    w = max(0.0, min(100.0, pct))
    return (f'<div class="row"><div class="rlab">{html.escape(label)}</div>'
            f'<div class="rbar"><div class="{cls}" style="width:{w:.1f}%"></div>'
            f'<span class="rpct">{pct:.1f}%{(" · " + html.escape(sub)) if sub else ""}</span>'
            f'</div></div>')


def _match_card(r):
    p1, pn, p2 = float(r["p_home_win"]), float(r["p_draw"]), float(r["p_away_win"])
    probs = {"1": p1, "N": pn, "2": p2}
    pred = max(probs, key=probs.get)
    home, away = html.escape(str(r["home_team"])), html.escape(str(r["away_team"]))
    grp = str(r.get("group", "")).replace("GROUP_", "")
    played = pd.notna(r["home_real"])

    if played:
        real = _outcome(r["home_real"], r["away_real"])
        ok = (pred == real)
        badge = (f'<span class="badge {"ok" if ok else "ko"}">'
                 f'{"✓" if ok else "✗"}</span>')
        score = (f'<div class="score real">{int(r["home_real"])}–{int(r["away_real"])}'
                 f'<span class="slab">réel</span></div>')
        state = "played"
    else:
        badge = '<span class="badge soon">à venir</span>'
        score = (f'<div class="score">{html.escape(str(r["pred_score"]))}'
                 f'<span class="slab">indicatif</span></div>')
        state = "upcoming"

    # barre empilée 1/N/2
    stack = (f'<div class="stack" title="V {p1*100:.0f}% / N {pn*100:.0f}% / D {p2*100:.0f}%">'
             f'<div class="seg s1" style="width:{p1*100:.1f}%"></div>'
             f'<div class="seg sn" style="width:{pn*100:.1f}%"></div>'
             f'<div class="seg s2" style="width:{p2*100:.1f}%"></div></div>')
    pick = {"1": home, "N": "Nul", "2": away}[pred]
    return (f'<div class="match {state}" data-group="{grp}" data-state="{state}">'
            f'<div class="mtop"><span class="grp">{grp}</span>{badge}</div>'
            f'<div class="teams"><span class="t">{home}</span>'
            f'<span class="vs">vs</span><span class="t">{away}</span></div>'
            f'{score}'
            f'{stack}'
            f'<div class="pick">Vainqueur prédit : <b>{html.escape(pick)}</b> '
            f'({max(p1,pn,p2)*100:.0f}%)</div></div>')


def generate_dashboard() -> str:
    matchs, champ, boot, fixtures = _load()
    m = _merge_actuals(matchs, fixtures)
    bilan = _compute_bilan(m)
    now = dt.datetime.now().strftime("%d/%m/%Y %H:%M")

    # ── KPIs ──
    if bilan["n"]:
        kpis = [
            ("Matchs joués", f'{bilan["n"]}'),
            ("Bons vainqueurs", f'{bilan["acc"]*100:.0f}%'),
            ("Score de Brier", f'{bilan["brier"]:.3f}'),
            ("Scores exacts", f'{bilan["exact"]*100:.0f}%'),
        ]
    else:
        kpis = [("Matchs joués", "0"), ("Bons vainqueurs", "—"),
                ("Score de Brier", "—"), ("Scores exacts", "—")]
    kpi_html = "".join(
        f'<div class="kpi"><div class="kv">{v}</div><div class="kl">{html.escape(l)}</div></div>'
        for l, v in kpis)

    # ── Bilan détaillé des matchs joués ──
    played_html = ""
    if bilan["n"]:
        rows = []
        for _, r in bilan["played"].iterrows():
            if pd.isna(r["p_home_win"]) or pd.isna(r["p_draw"]) or pd.isna(r["p_away_win"]):
                continue
            probs = {"1": float(r["p_home_win"]), "N": float(r["p_draw"]), "2": float(r["p_away_win"])}
            pred = max(probs, key=probs.get)
            real = _outcome(r["home_real"], r["away_real"])
            ok = pred == real
            pick = {"1": r["home_team"], "N": "Nul", "2": r["away_team"]}[pred]
            rows.append(
                f'<tr class="{"ok" if ok else "ko"}">'
                f'<td>{html.escape(str(r["home_team"]))} – {html.escape(str(r["away_team"]))}</td>'
                f'<td class="c">{html.escape(str(pick))} ({max(probs.values())*100:.0f}%)</td>'
                f'<td class="c"><b>{int(r["home_real"])}–{int(r["away_real"])}</b></td>'
                f'<td class="c">{"✓" if ok else "✗"}</td></tr>')
        played_html = (
            '<p class="legend">✓ = <b>vainqueur</b> correctement prédit. Le score exact '
            '(souvent « 1-1 » indicatif) n\'entre PAS dans ce jugement : il est ~10% '
            'prévisible pour n\'importe quel modèle.</p>'
            '<table class="tbl"><thead><tr><th>Match</th><th>Pronostic (vainqueur)</th>'
            '<th>Score réel</th><th>Bon&nbsp;?</th></tr></thead><tbody>'
            + "".join(rows) + "</tbody></table>")

    # ── Favoris au titre ──
    champ_html = ""
    if not champ.empty:
        top = champ.head(14)
        mx = float(top["win_prob"].max()) * 100 if "win_prob" in top else 100
        for _, r in top.iterrows():
            pct = float(r["win_prob"]) * 100
            champ_html += _bar_row(str(r["team"]), pct, cls="bar")

    # ── Soulier d'or ──
    boot_html = ""
    if not boot.empty and "golden_boot_prob" in boot:
        for _, r in boot.head(12).iterrows():
            pct = float(r["golden_boot_prob"]) * 100
            boot_html += _bar_row(f'{r["player"]} ({r["team"]})', pct, cls="bar gold")

    # ── Matchs (groupés) ──
    groups = sorted({str(x).replace("GROUP_", "") for x in m["group"].dropna()}) if not m.empty else []
    filt = '<button class="f active" data-f="all">Tous</button>'
    filt += '<button class="f" data-f="played">Joués</button>'
    for g in groups:
        filt += f'<button class="f" data-f="g{g}">{g}</button>'
    cards = "".join(_match_card(r) for _, r in m.iterrows()) if not m.empty else "<p>Aucun match.</p>"

    return _TEMPLATE.format(
        now=now, kpis=kpi_html,
        played_section=(f'<section class="card"><h2>Bilan des matchs joués</h2>{played_html}</section>'
                        if played_html else ""),
        champ=champ_html or "<p>Lance predict.py d'abord.</p>",
        boot=boot_html or "<p>—</p>",
        filters=filt, matches=cards)


_TEMPLATE = """<!DOCTYPE html>
<html lang="fr"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Pronostic CdM 2026 — Dashboard</title>
<style>
:root{{--bg:#0f1116;--card:#171a21;--card2:#1e222b;--line:#2a2f3a;--tx:#e7e9ee;
--mut:#9aa3b2;--s1:#3b82f6;--sn:#6b7280;--s2:#f97316;--ok:#22c55e;--ko:#ef4444;--gold:#eab308;}}
*{{box-sizing:border-box}}
body{{margin:0;background:var(--bg);color:var(--tx);font:15px/1.5 system-ui,Segoe UI,Roboto,sans-serif}}
.wrap{{max-width:1180px;margin:0 auto;padding:24px}}
header{{display:flex;justify-content:space-between;align-items:baseline;flex-wrap:wrap;gap:8px;margin-bottom:18px}}
h1{{font-size:24px;margin:0}}
h2{{font-size:16px;margin:0 0 14px;color:var(--tx)}}
.sub{{color:var(--mut);font-size:13px}}
.kpis{{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin-bottom:18px}}
.kpi{{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:16px;text-align:center}}
.kv{{font-size:28px;font-weight:700}}
.kl{{color:var(--mut);font-size:12px;margin-top:2px}}
.grid{{display:grid;grid-template-columns:1fr 1fr;gap:16px}}
.card{{background:var(--card);border:1px solid var(--line);border-radius:14px;padding:18px;margin-bottom:16px}}
.row{{display:flex;align-items:center;gap:10px;margin:7px 0}}
.rlab{{width:140px;font-size:13px;color:var(--tx);text-align:right;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}}
.rbar{{flex:1;background:var(--card2);border-radius:7px;height:22px;position:relative;overflow:hidden}}
.bar{{height:100%;background:var(--s1);border-radius:7px}}
.bar.gold{{background:var(--gold)}}
.rpct{{position:absolute;left:8px;top:0;line-height:22px;font-size:12px;color:#fff;font-weight:600}}
.filters{{display:flex;gap:6px;flex-wrap:wrap;margin-bottom:14px}}
.f{{background:var(--card2);border:1px solid var(--line);color:var(--mut);border-radius:8px;
padding:5px 11px;font-size:13px;cursor:pointer}}
.f.active{{background:var(--s1);color:#fff;border-color:var(--s1)}}
.matches{{display:grid;grid-template-columns:repeat(auto-fill,minmax(225px,1fr));gap:12px}}
.match{{background:var(--card2);border:1px solid var(--line);border-radius:12px;padding:12px}}
.match.played{{border-color:#3a4150}}
.mtop{{display:flex;justify-content:space-between;align-items:center;margin-bottom:6px}}
.grp{{font-size:11px;color:var(--mut);font-weight:700;letter-spacing:.5px}}
.badge{{font-size:11px;padding:2px 7px;border-radius:20px;font-weight:700}}
.badge.ok{{background:rgba(34,197,94,.15);color:var(--ok)}}
.badge.ko{{background:rgba(239,68,68,.15);color:var(--ko)}}
.badge.soon{{background:var(--card);color:var(--mut)}}
.teams{{display:flex;align-items:center;gap:6px;font-weight:600;font-size:14px}}
.teams .vs{{color:var(--mut);font-size:11px;font-weight:400}}
.teams .t{{flex:1}}.teams .t:last-child{{text-align:right}}
.score{{font-size:22px;font-weight:700;text-align:center;margin:8px 0 2px}}
.score.real{{color:var(--ok)}}
.slab{{display:block;font-size:10px;color:var(--mut);font-weight:400}}
.stack{{display:flex;height:8px;border-radius:5px;overflow:hidden;margin:8px 0 6px}}
.seg.s1{{background:var(--s1)}}.seg.sn{{background:var(--sn)}}.seg.s2{{background:var(--s2)}}
.pick{{font-size:12px;color:var(--mut)}}.pick b{{color:var(--tx)}}
.tbl{{width:100%;border-collapse:collapse;font-size:13px}}
.tbl th{{text-align:left;color:var(--mut);font-weight:600;padding:6px 8px;border-bottom:1px solid var(--line)}}
.tbl td{{padding:7px 8px;border-bottom:1px solid var(--line)}}
.tbl td.c{{text-align:center}}
.tbl tr.ok td:last-child{{color:var(--ok)}}.tbl tr.ko td:last-child{{color:var(--ko)}}
.legend{{font-size:12px;color:var(--mut);margin-top:8px}}
.legend b.s1{{color:var(--s1)}}.legend b.sn{{color:var(--sn)}}.legend b.s2{{color:var(--s2)}}
@media(max-width:820px){{.grid{{grid-template-columns:1fr}}.kpis{{grid-template-columns:repeat(2,1fr)}}}}
</style></head><body><div class="wrap">
<header>
  <div><h1>⚽ Pronostic Coupe du Monde 2026</h1>
  <div class="sub">Tableau de bord — mis à jour le {now}</div></div>
</header>

<div class="kpis">{kpis}</div>

{played_section}

<div class="grid">
  <section class="card"><h2>🏆 Favoris au titre</h2>{champ}</section>
  <section class="card"><h2>👟 Soulier d'or</h2>{boot}</section>
</div>

<section class="card">
  <h2>📅 Tous les matchs</h2>
  <div class="filters">{filters}</div>
  <div class="matches" id="matches">{matches}</div>
  <div class="legend">Barre : <b class="s1">Victoire</b> · <b class="sn">Nul</b> ·
  <b class="s2">Défaite</b>. Le « score indicatif » est le score le plus probable
  (le score exact est ~10% prévisible — juger sur le pronostic 1/N/2).</div>
</section>

<script>
document.querySelectorAll('.f').forEach(function(b){{
  b.onclick=function(){{
    document.querySelectorAll('.f').forEach(x=>x.classList.remove('active'));
    b.classList.add('active');
    var f=b.dataset.f;
    document.querySelectorAll('.match').forEach(function(m){{
      var show = f==='all' || (f==='played'&&m.dataset.state==='played') || ('g'+m.dataset.group===f);
      m.style.display = show ? '' : 'none';
    }});
  }};
}});
</script>
</div></body></html>"""


def main():
    htmltext = generate_dashboard()
    OUT.write_text(htmltext, encoding="utf-8")
    print(f"Dashboard genere -> {OUT}")
    return OUT


if __name__ == "__main__":
    main()
