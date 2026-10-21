#!/usr/bin/env python3
"""Probabilidade de empate dos jogos do dia: Liga Portugal, La Liga e Serie A.

Uso:
  export FOOTBALL_DATA_TOKEN=...        (chave grátis em football-data.org)
  python3 jogos_do_dia.py               (jogos de hoje, hora de Lisboa)
  python3 jogos_do_dia.py --data 2026-10-25
Opcional: TELEGRAM_TOKEN e TELEGRAM_CHAT_ID para receber o resultado no Telegram.
Só usa a biblioteca standard do Python (3.9+).
"""
import argparse, html, json, math, os, sys, time, unicodedata, urllib.request
from collections import deque
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

ROOT = os.path.dirname(os.path.abspath(__file__))
MODELO = json.load(open(os.path.join(ROOT, "modelo.json")))
LISBOA = ZoneInfo("Europe/Lisbon")
LIGAS = {"PPL": ("Liga Portugal", "PT"), "PD": ("La Liga", "ES"), "SA": ("Serie A", "IT")}
NOMES = {"PT": "Liga Portugal", "ES": "La Liga", "IT": "Serie A"}
LIMITE_POSICOES = 6      # diferença máxima de posições na tabela
TOLERANCIA_PONTOS = 0     # também entram equipas empatadas em pontos com a equipa que está no limite das 6 posições
MIN_JOGOS = 5            # jogos mínimos de cada equipa na época


def norm(s):
    s = unicodedata.normalize("NFKD", s or "").encode("ascii", "ignore").decode().lower()
    return s.replace(".", " ")


# ---------------------------------------------------------------- elegibilidade
def _braga(n): return "braga" in n
def _grande_pt(n):
    if _braga(n): return False
    t = n.split()
    return "benfica" in n or "sporting" in t or "sp lisbon" in n or "porto" in t
def _grande_es(n):
    return "real madrid" in n or ("barcelona" in n and "espanyol" not in n) or ("atletico" in n and "madrid" in n) or "ath madrid" in n

def elegivel(liga, casa, fora):
    """Regras do Pedro: PT/ES excluem jogos de um grande contra equipa fora do grupo; IT sem exclusões."""
    c, f = norm(casa), norm(fora)
    if liga == "PT":
        if _grande_pt(c) or _grande_pt(f):
            return all(_grande_pt(x) or _braga(x) for x in (c, f))
        return True
    if liga == "ES":
        return _grande_es(c) == _grande_es(f)
    return True


# ---------------------------------------------------------------- características
def _stats(q):
    l = list(q)
    if len(l) < 8: return None
    l10 = l[-10:]
    m = lambda xs: sum(xs) / len(xs)
    return dict(tg=m([x[0] + x[1] for x in l10]), ga=m([x[1] for x in l10]), gf=m([x[0] for x in l10]),
                dr=m([x[2] for x in l]), pt=m([x[3] for x in l10]))


def probabilidade(f):
    z = MODELO["intercept"]
    for c, mu, sd, b in zip(MODELO["cols"], MODELO["mean"], MODELO["scale"], MODELO["coef"]):
        z += b * (f[c] - mu) / sd
    return 1 / (1 + math.exp(-z))


def escalao(p):
    return "ALTA" if p >= MODELO.get("corte_alta", 0.315) else "média" if p >= MODELO.get("corte_media", 0.288) else "baixa"


def analisar(partidas, liga, dia, epoca_inicio):
    """partidas: lista de dicts {data (datetime com fuso), casa, fora, gc, gf (None se por jogar)}.
    dia: date (hora de Lisboa). epoca_inicio: datetime em que começou a época atual.
    Usa apenas jogos terminados antes de `dia`."""
    inicio_dia = datetime(dia.year, dia.month, dia.day, tzinfo=LISBOA)
    feitos = sorted((p for p in partidas if p["gc"] is not None and p["data"] < inicio_dia), key=lambda p: p["data"])
    hist = {}; tab = {}
    for p in feitos:
        gc, gf = p["gc"], p["gf"]
        pc, pf = (3, 0) if gc > gf else (0, 3) if gc < gf else (1, 1)
        emp = int(gc == gf)
        hist.setdefault(p["casa"], deque(maxlen=20)).append((gc, gf, emp, pc))
        hist.setdefault(p["fora"], deque(maxlen=20)).append((gf, gc, emp, pf))
        if p["data"] >= epoca_inicio:
            for t, pts, gm, gs in ((p["casa"], pc, gc, gf), (p["fora"], pf, gf, gc)):
                r = tab.setdefault(t, [0, 0, 0, 0]); r[0] += pts; r[1] += gm - gs; r[2] += gm; r[3] += 1
    ordem = sorted(tab, key=lambda t: (-tab[t][0], -tab[t][1], -tab[t][2], t)); pos = {t: i + 1 for i, t in enumerate(ordem)}; pts_pos = [tab[t][0] for t in ordem]
    linhas = []
    for p in partidas:
        if p["gc"] is not None or p["data"].astimezone(LISBOA).date() != dia: continue
        c, f = p["casa"], p["fora"]
        L = dict(liga=liga, hora=p["data"].astimezone(LISBOA), casa=c, fora=f, elegivel=elegivel(liga, c, f), p=None, filtro=False, nota="")
        sh, sa = _stats(hist.get(c, [])), _stats(hist.get(f, []))
        if c not in pos or f not in pos or sh is None or sa is None:
            L["nota"] = "sem histórico suficiente"; linhas.append(L); continue
        L.update(pos_casa=pos[c], pos_fora=pos[f], jogos=min(tab[c][3], tab[f][3]))
        feat = dict(dpos=abs(pos[c] - pos[f]), formad=abs(sh["pt"] - sa["pt"]), golos=(sh["tg"] + sa["tg"]) / 2,
                    tend=(sh["dr"] + sa["dr"]) / 2, marcam=(sh["gf"] + sa["gf"]) / 2, sofrem=(sh["ga"] + sa["ga"]) / 2)
        L["feat"] = feat; L["p"] = probabilidade(feat)
        # perto na tabela: ≤6 posições OU a equipa de baixo tem os mesmos pontos da que fica no limite (e vice-versa)
        t, b = min(pos[c], pos[f]), max(pos[c], pos[f]); n = len(pts_pos)
        p_top, p_bot = pts_pos[t - 1], pts_pos[b - 1]
        limite_baixo, limite_cima = pts_pos[min(t + LIMITE_POSICOES, n) - 1], pts_pos[max(b - LIMITE_POSICOES, 1) - 1]
        perto_pos = feat["dpos"] <= LIMITE_POSICOES
        perto_pts = p_bot >= limite_baixo - TOLERANCIA_PONTOS or p_top <= limite_cima + TOLERANCIA_PONTOS
        L["por_tolerancia"] = (not perto_pos) and perto_pts
        L["filtro"] = L["elegivel"] and (perto_pos or perto_pts) and L["jogos"] >= MIN_JOGOS
        if not L["elegivel"]: L["nota"] = "jogo de grande contra equipa fora do grupo: fora da estratégia"
        elif L["jogos"] < MIN_JOGOS: L["nota"] = f"menos de {MIN_JOGOS} jogos na época"
        elif not (perto_pos or perto_pts): L["nota"] = f"{feat['dpos']} posições e {abs(p_top - p_bot)} pontos de diferença: longe na tabela"
        elif L["por_tolerancia"]: L["nota"] = f"incluído por tolerância de pontos ({feat['dpos']} posições, {abs(p_top - p_bot)} pts)"
        linhas.append(L)
    return linhas


# ---------------------------------------------------------------- apresentação
def texto(linhas, dia):
    out = [f"Jogos de {dia:%d/%m/%Y}  (P = probabilidade de empate estimada; média histórica {MODELO['taxa']:.0%})", ""]
    if not linhas: return "\n".join(out + ["Sem jogos nas 3 ligas."])
    for L in sorted(linhas, key=lambda x: (not x["filtro"], x["hora"], x["liga"])):
        marca = "✅" if L["filtro"] else "  "
        if L["p"] is None:
            out.append(f"{marca} {L['hora']:%H:%M} {NOMES[L['liga']]}: {L['casa']} - {L['fora']} | {L['nota']}")
        else:
            extra = ""
            out.append(f"{marca} {L['hora']:%H:%M} {NOMES[L['liga']]}: {L['casa']} ({L['pos_casa']}º) - {L['fora']} ({L['pos_fora']}º) | P {L['p']:.1%} {escalao(L['p'])} | marcam {L['feat']['marcam']:.2f} golos/jogo{extra}" + (f" | {L['nota']}" if L["nota"] else ""))
    out += ["", "✅ = jogo elegível, ≥5 jogos de cada equipa e equipas perto na tabela (≤6 posições, ou empatadas em pontos com o limite das 6 posições).",
            "Escala: ALTA ≥31,5% (históricos ✅: ~33% empates) | média 28,8–31,5% (~31%) | baixa <28,8% (~28%).", "Estimativa com pouco poder de previsão."]
    return "\n".join(out)


def pagina(linhas, dia):
    rows = ""
    for L in sorted(linhas, key=lambda x: (not x["filtro"], x["hora"], x["liga"])):
        p = "-" if L["p"] is None else f"{L['p']:.1%}"
        esc = "" if L["p"] is None else escalao(L["p"])
        pos = "" if L["p"] is None else f"{L['pos_casa']}º vs {L['pos_fora']}º"
        gm = "" if L["p"] is None else f"{L['feat']['marcam']:.2f}"
        rows += (f"<tr class='{'ok' if L['filtro'] else ''}'><td>{L['hora']:%H:%M}</td><td>{html.escape(NOMES[L['liga']])}</td>"
                 f"<td>{html.escape(L['casa'])} - {html.escape(L['fora'])}</td><td>{pos}</td><td><b>{p}</b> <span class='tag {esc}'>{esc}</span></td><td>{gm}</td>"
                 f"<td>{'✅' if L['filtro'] else ''}</td><td>{html.escape(L['nota'])}</td></tr>")
    return f"""<!doctype html><html lang="pt"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Jogos de {dia:%d/%m}</title><style>body{{font-family:system-ui;margin:16px;max-width:900px}}table{{border-collapse:collapse;width:100%}}
td,th{{padding:6px 8px;border-bottom:1px solid #ddd;text-align:left;font-size:14px}}tr.ok{{background:#e8f6ec}}small{{color:#666}}.tag{{display:inline-block;padding:1px 8px;border-radius:10px;font-size:12px;font-weight:600}}.ALTA{{background:#1b8a3a;color:#fff}}.média{{background:#f0b429;color:#222}}.baixa{{background:#d5d9de;color:#333}}.leg{{border:1px solid #ddd;border-radius:8px;padding:10px 12px;margin:16px 0 8px;font-size:14px}}.leg div{{margin:4px 0}}</style></head><body>
<h2>Jogos de {dia:%d/%m/%Y}</h2><table><tr><th>Hora</th><th>Liga</th><th>Jogo</th><th>Tabela</th><th>P(empate)</th><th>Golos marcados/jogo</th><th>Filtro</th><th>Nota</th></tr>{rows}</table>
<div class="leg"><b>Escala da probabilidade de empate</b>
<div><span class="tag ALTA">ALTA</span> 31,5% ou mais &nbsp;·&nbsp; nos jogos históricos com ✅ empataram ~33%</div>
<div><span class="tag média">média</span> 28,8% a 31,5% &nbsp;·&nbsp; empataram ~31%</div>
<div><span class="tag baixa">baixa</span> menos de 28,8% &nbsp;·&nbsp; empataram ~28%</div>
<div style="margin-top:8px"><b>Golos marcados/jogo:</b> média de golos marcados por jogo pelas duas equipas nos últimos 10 jogos de cada uma. Quanto menos golos marcam, mais empates: abaixo de 1,2 → ~31–32%; acima de 1,45 → ~26%.</div>
<div><small>✅ = jogo elegível, ≥{MIN_JOGOS} jogos de cada equipa e equipas perto na tabela (≤{LIMITE_POSICOES} posições de diferença, ou os mesmos pontos da equipa que fica no limite). Estimativa com pouco poder de previsão.</small></div></div>
<p><small>Use a probabilidade como filtro, não como certeza.</small></p></body></html>"""


# ---------------------------------------------------------------- API
def api(path, token):
    req = urllib.request.Request("https://api.football-data.org/v4" + path, headers={"X-Auth-Token": token})
    with urllib.request.urlopen(req, timeout=30) as r: return json.load(r)


def descarregar(token, dia):
    ano = dia.year if dia.month >= 7 else dia.year - 1
    res = {}
    for code, (nome, lg) in LIGAS.items():
        ps = []
        for epoca in (ano - 1, ano):
            time.sleep(7)  # plano gratuito: 10 chamadas por minuto
            for m in api(f"/competitions/{code}/matches?season={epoca}", token)["matches"]:
                if not m["homeTeam"].get("name") or not m["awayTeam"].get("name"): continue
                ft = m["score"]["fullTime"]; fin = m["status"] == "FINISHED" and ft["home"] is not None
                ps.append(dict(data=datetime.fromisoformat(m["utcDate"].replace("Z", "+00:00")), casa=m["homeTeam"]["name"], fora=m["awayTeam"]["name"],
                               gc=ft["home"] if fin else None, gf=ft["away"] if fin else None, estado=m["status"]))
        res[code] = (ps, datetime(ano, 7, 1, tzinfo=LISBOA))
    return res


def telegram(msg):
    tk, ch = os.environ.get("TELEGRAM_TOKEN"), os.environ.get("TELEGRAM_CHAT_ID")
    if not (tk and ch): return
    data = json.dumps({"chat_id": ch, "text": msg[:4000]}).encode()
    urllib.request.urlopen(urllib.request.Request(f"https://api.telegram.org/bot{tk}/sendMessage", data=data, headers={"Content-Type": "application/json"}), timeout=30)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--data"); ap.add_argument("--saida", default=os.path.join(ROOT, "jogos_do_dia.html")); a = ap.parse_args()
    dia = datetime.strptime(a.data, "%Y-%m-%d").date() if a.data else datetime.now(LISBOA).date()
    token = os.environ.get("FOOTBALL_DATA_TOKEN")
    if not token: sys.exit("Falta a variável FOOTBALL_DATA_TOKEN (chave grátis em football-data.org).")
    dados = descarregar(token, dia); linhas = []
    for code, (ps, ini) in dados.items(): linhas += analisar(ps, LIGAS[code][1], dia, ini)
    t = texto(linhas, dia); print(t)
    os.makedirs(os.path.dirname(os.path.abspath(a.saida)), exist_ok=True)
    open(a.saida, "w", encoding="utf-8").write(pagina(linhas, dia)); telegram(t)


if __name__ == "__main__":
    main()
