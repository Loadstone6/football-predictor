const OUTCOMES = [
  { key: "home", label: "Home" },
  { key: "draw", label: "Draw" },
  { key: "away", label: "Away" },
];

let report = null;
let currentMatch = null;

const el = (id) => document.getElementById(id);
const clip = (value, min, max) => Math.max(min, Math.min(max, value));
const fmtPct = (value) => `${(Number(value || 0) * 100).toFixed(1)}%`;
const fmtSignedPct = (value) => `${Number(value || 0) >= 0 ? "+" : ""}${fmtPct(value)}`;
const fmtNum = (value, digits = 2) => Number(value || 0).toFixed(digits);
const fmtMoney = (value) => `${Number(value || 0) >= 0 ? "" : "-"}£${Math.abs(Number(value || 0)).toFixed(2)}`;

function renderMetricCards(summary) {
  const betting = summary.betting || {};
  const metrics = [
    ["Matches", summary.matches, 0],
    ["Accuracy", summary.accuracy, 3],
    ["Log loss", summary.log_loss, 3],
    ["Brier", summary.brier, 3],
    ["ROI", betting.roi, 3],
    ["Max DD", betting.max_drawdown, 3],
  ];
  el("metricCards").innerHTML = metrics
    .map(([label, value, digits]) => `
      <div class="metric-card">
        <span>${label}</span>
        <strong>${digits === 0 ? Number(value || 0).toFixed(0) : fmtNum(value, digits)}</strong>
      </div>
    `)
    .join("");
}

function renderMatchSelector() {
  const select = el("matchSelect");
  const matches = report.predictions || [];
  select.innerHTML = matches
    .map((match, index) => {
      const label = `${match.date}  ${match.home_team} vs ${match.away_team}`;
      return `<option value="${index}">${label}</option>`;
    })
    .join("");
  select.value = String(Math.max(0, matches.length - 1));
  select.addEventListener("change", () => {
    currentMatch = matches[Number(select.value)];
    renderMatch(currentMatch);
  });
  currentMatch = matches[matches.length - 1] || report.latest_match;
}

function outcomeObjectFromMatch(match) {
  return {
    home: Number(match.p_home || 0),
    draw: Number(match.p_draw || 0),
    away: Number(match.p_away || 0),
  };
}

function renderOutcomeBars(containerId, probabilities) {
  el(containerId).innerHTML = OUTCOMES.map(({ key, label }) => {
    const value = Number(probabilities[key] || 0);
    return `
      <div class="prob-row">
        <span>${label}</span>
        <div class="bar-track"><div class="bar-fill" style="width:${Math.max(2, value * 100)}%"></div></div>
        <span>${fmtPct(value)}</span>
      </div>
    `;
  }).join("");
}

function renderHeatmapInto(containerId, matrix) {
  const safeMatrix = matrix || [];
  const max = Math.max(...safeMatrix.flat().map(Number), 0.0001);
  let html = "";
  safeMatrix.forEach((row, homeGoals) => {
    row.forEach((value, awayGoals) => {
      const intensity = Math.max(0.08, Number(value) / max);
      const color = `rgba(24, 123, 82, ${0.25 + 0.75 * intensity})`;
      html += `<div class="heat-cell" style="background:${color}" title="${homeGoals}-${awayGoals}: ${fmtPct(value)}">${homeGoals}-${awayGoals}</div>`;
    });
  });
  el(containerId).innerHTML = html;
}

function noVig(odds) {
  if (odds.some((value) => !value || Number(value) <= 1)) return null;
  const implied = odds.map((value) => 1 / Number(value));
  const total = implied.reduce((a, b) => a + b, 0);
  return implied.map((value) => value / total);
}

function kellyFraction(probability, odds) {
  const b = Number(odds) - 1;
  if (b <= 0) return 0;
  return Math.max(0, ((b * probability) - (1 - probability)) / b);
}

function valueRows(probabilities, odds, bankroll) {
  const market = noVig(odds);
  if (!market) return null;
  const bettingConfig = report.betting_config || {};
  const maxStakeFraction = Number(bettingConfig.max_stake_fraction ?? 0.03);
  const fractionalKelly = Number(bettingConfig.fractional_kelly ?? 0.25);
  return OUTCOMES.map(({ key, label }, index) => {
    const model = Number(probabilities[key] || 0);
    const odd = Number(odds[index]);
    const edge = model - market[index];
    const ev = model * odd - 1;
    const kelly = kellyFraction(model, odd);
    const stake = Number(bankroll || 0) * Math.min(maxStakeFraction, fractionalKelly * kelly);
    return { key, label, model, market: market[index], odds: odd, edge, ev, stake };
  });
}

function renderValueTable(containerId, probabilities, odds, bankroll) {
  const rows = valueRows(probabilities, odds, bankroll);
  if (!rows) {
    el(containerId).innerHTML = `<div class="muted">No odds on this row</div>`;
    return;
  }
  el(containerId).innerHTML = rows.map((row) => {
    const pillClass = row.edge > 0.04 ? "pill-good" : row.edge < -0.04 ? "pill-bad" : "pill-neutral";
    return `
      <div class="value-row">
        <div>
          <strong>${row.label}</strong>
          <div class="muted">model ${fmtPct(row.model)} / market ${fmtPct(row.market)} / EV ${fmtSignedPct(row.ev)}</div>
        </div>
        <span class="${pillClass}">${fmtSignedPct(row.edge)} · ${fmtMoney(row.stake)}</span>
      </div>
    `;
  }).join("");
}

function renderMatch(match) {
  if (!match) return;
  el("homeTeam").textContent = match.home_team || "-";
  el("awayTeam").textContent = match.away_team || "-";
  el("actualScore").textContent = `${Number(match.home_goals).toFixed(0)}-${Number(match.away_goals).toFixed(0)}`;
  el("topScore").textContent = match.top_score || "-";
  renderOutcomeBars("probBars", outcomeObjectFromMatch(match));
  renderHeatmapInto("heatmap", match.score_matrix || report.score_matrix);
  renderValueTable(
    "valueTable",
    outcomeObjectFromMatch(match),
    [match.home_odds, match.draw_odds, match.away_odds],
    report.summary?.betting?.ending_bankroll || 1000
  );
}

function teamByName(name) {
  return (report.team_profiles || []).find((team) => team.team === name);
}

function poissonPmf(lambda, maxGoals) {
  const values = [];
  let factorial = 1;
  for (let k = 0; k <= maxGoals; k += 1) {
    if (k > 0) factorial *= k;
    values.push(Math.exp(-lambda) * Math.pow(lambda, k) / factorial);
  }
  const total = values.reduce((a, b) => a + b, 0);
  return values.map((value) => value / total);
}

function poissonMatrix(homeXg, awayXg, maxGoals = 5) {
  const home = poissonPmf(clip(homeXg, 0.05, 6), maxGoals);
  const away = poissonPmf(clip(awayXg, 0.05, 6), maxGoals);
  return home.map((homeValue) => away.map((awayValue) => homeValue * awayValue));
}

function scoreOutcomeProbabilities(matrix) {
  let home = 0;
  let draw = 0;
  let away = 0;
  matrix.forEach((row, h) => {
    row.forEach((value, a) => {
      if (h > a) home += value;
      else if (h === a) draw += value;
      else away += value;
    });
  });
  const total = home + draw + away;
  return { home: home / total, draw: draw / total, away: away / total };
}

function topScore(matrix) {
  let best = { score: "0-0", probability: 0 };
  matrix.forEach((row, h) => {
    row.forEach((value, a) => {
      if (value > best.probability) best = { score: `${h}-${a}`, probability: value };
    });
  });
  return best;
}

function eloThreeWay(homeElo, awayElo, neutral) {
  const homeAdjustment = neutral ? 0 : 55;
  const diff = homeElo + homeAdjustment - awayElo;
  const binaryHome = 1 / (1 + Math.pow(10, -diff / 400));
  const draw = clip(0.08 + 0.26 * Math.exp(-Math.abs(diff) / 420), 0.08, 0.34);
  const home = (1 - draw) * binaryHome;
  const away = (1 - draw) * (1 - binaryHome);
  const total = home + draw + away;
  return { home: home / total, draw: draw / total, away: away / total };
}

function labPrediction(homeTeam, awayTeam, neutral) {
  const elo = eloThreeWay(homeTeam.elo, awayTeam.elo, neutral);
  const eloDiff = Number(homeTeam.elo || 1500) - Number(awayTeam.elo || 1500);
  const homeXg = clip(
    0.55 * Number(homeTeam.attack_form || 1.25)
      + 0.25 * Number(awayTeam.defense_form || 1.25)
      + 0.20 * Number(homeTeam.xg_form || 1.25)
      + (neutral ? 0 : 0.12)
      + 0.0018 * eloDiff,
    0.25,
    3.75
  );
  const awayXg = clip(
    0.55 * Number(awayTeam.attack_form || 1.25)
      + 0.25 * Number(homeTeam.defense_form || 1.25)
      + 0.20 * Number(awayTeam.xg_form || 1.25)
      - 0.0012 * eloDiff,
    0.25,
    3.75
  );
  const matrix = poissonMatrix(homeXg, awayXg, 5);
  const scoreProbs = scoreOutcomeProbabilities(matrix);
  const probabilities = {
    home: 0.62 * elo.home + 0.38 * scoreProbs.home,
    draw: 0.62 * elo.draw + 0.38 * scoreProbs.draw,
    away: 0.62 * elo.away + 0.38 * scoreProbs.away,
  };
  const total = probabilities.home + probabilities.draw + probabilities.away;
  probabilities.home /= total;
  probabilities.draw /= total;
  probabilities.away /= total;
  return { probabilities, matrix, topScore: topScore(matrix), homeXg, awayXg };
}

function populateLabTeams() {
  const teams = report.team_profiles || [];
  const options = teams.map((team) => `<option value="${team.team}">${team.rank}. ${team.team}</option>`).join("");
  el("labHomeTeam").innerHTML = options;
  el("labAwayTeam").innerHTML = options;
  if (teams[0]) el("labHomeTeam").value = teams[0].team;
  if (teams[1]) el("labAwayTeam").value = teams[1].team;
  ["labHomeTeam", "labAwayTeam", "labNeutral", "labHomeOdds", "labDrawOdds", "labAwayOdds", "labBankroll"].forEach((id) => {
    el(id).addEventListener("input", renderLab);
    el(id).addEventListener("change", renderLab);
  });
}

function renderTeamCompare(home, away) {
  const rows = [
    { label: "Elo", home: home.elo, away: away.elo, scoreHome: home.elo, scoreAway: away.elo, digits: 0 },
    { label: "Power", home: home.power_score, away: away.power_score, scoreHome: home.power_score, scoreAway: away.power_score, digits: 1 },
    { label: "PPM", home: home.points_per_match, away: away.points_per_match, scoreHome: home.points_per_match, scoreAway: away.points_per_match, digits: 2 },
    { label: "xG diff", home: home.xg_diff_per_match, away: away.xg_diff_per_match, scoreHome: home.xg_diff_per_match + 2, scoreAway: away.xg_diff_per_match + 2, digits: 2 },
    { label: "Attack", home: home.goals_for_per_match, away: away.goals_for_per_match, scoreHome: home.goals_for_per_match, scoreAway: away.goals_for_per_match, digits: 2 },
    { label: "Defence", home: home.goals_against_per_match, away: away.goals_against_per_match, scoreHome: 4 - home.goals_against_per_match, scoreAway: 4 - away.goals_against_per_match, digits: 2 },
  ];
  el("teamCompare").innerHTML = rows.map((row) => {
    const homeScore = Math.max(0, row.scoreHome);
    const awayScore = Math.max(0, row.scoreAway);
    const total = homeScore + awayScore || 1;
    const homeWidth = (homeScore / total) * 100;
    const awayWidth = (awayScore / total) * 100;
    return `
      <div class="compare-row">
        <strong>${fmtNum(row.home, row.digits)}</strong>
        <div>
          <div class="muted">${row.label}</div>
          <div class="compare-track">
            <span style="left:0;width:${homeWidth}%"></span>
            <span class="away" style="width:${awayWidth}%"></span>
          </div>
        </div>
        <strong>${fmtNum(row.away, row.digits)}</strong>
      </div>
    `;
  }).join("");
}

function renderLab() {
  const home = teamByName(el("labHomeTeam").value);
  const away = teamByName(el("labAwayTeam").value);
  if (!home || !away) return;
  const prediction = labPrediction(home, away, el("labNeutral").checked);
  renderOutcomeBars("labProbBars", prediction.probabilities);
  renderHeatmapInto("labHeatmap", prediction.matrix);
  renderTeamCompare(home, away);
  renderValueTable(
    "labValueTable",
    prediction.probabilities,
    [el("labHomeOdds").value, el("labDrawOdds").value, el("labAwayOdds").value],
    el("labBankroll").value
  );
}

function renderRankList(id, rows, valueAccessor) {
  el(id).innerHTML = (rows || []).map((row) => {
    const value = valueAccessor(row);
    return `
      <div class="rank-row">
        <div>
          <strong>${row.label || row.feature}</strong>
          <div class="muted">${row.feature}</div>
        </div>
        <span>${value}</span>
      </div>
    `;
  }).join("");
}

function pointsFromRows(rows, field, width, height, pad) {
  const values = rows.map((row) => Number(row[field] || 0));
  const min = Math.min(...values);
  const max = Math.max(...values);
  const span = Math.max(1e-9, max - min);
  return values.map((value, index) => {
    const x = pad + (index / Math.max(1, values.length - 1)) * (width - pad * 2);
    const y = height - pad - ((value - min) / span) * (height - pad * 2);
    return [x, y];
  });
}

function renderLineChart(id, rows, field, className) {
  const svg = el(id);
  if (!rows || rows.length === 0) {
    svg.innerHTML = "";
    return;
  }
  const width = 720;
  const height = 260;
  const pad = 28;
  const points = pointsFromRows(rows, field, width, height, pad);
  const d = points.map(([x, y], index) => `${index === 0 ? "M" : "L"} ${x.toFixed(1)} ${y.toFixed(1)}`).join(" ");
  svg.innerHTML = `
    <line class="axis" x1="${pad}" y1="${height - pad}" x2="${width - pad}" y2="${height - pad}"></line>
    <line class="axis" x1="${pad}" y1="${pad}" x2="${pad}" y2="${height - pad}"></line>
    <path class="${className}" d="${d}"></path>
  `;
}

function renderCalibration() {
  const rows = report.calibration || [];
  const svg = el("calibrationChart");
  const width = 720;
  const height = 260;
  const pad = 32;
  const bars = rows.map((row, index) => {
    const x = pad + index * ((width - pad * 2) / Math.max(1, rows.length));
    const barWidth = Math.max(20, (width - pad * 2) / Math.max(1, rows.length) - 10);
    const confidenceY = height - pad - Number(row.mean_confidence) * (height - pad * 2);
    const accuracyY = height - pad - Number(row.accuracy) * (height - pad * 2);
    return `
      <rect x="${x}" y="${accuracyY}" width="${barWidth}" height="${height - pad - accuracyY}" fill="#187b52" opacity="0.72"></rect>
      <circle cx="${x + barWidth / 2}" cy="${confidenceY}" r="4" class="chart-dot"></circle>
    `;
  }).join("");
  svg.innerHTML = `
    <line class="axis" x1="${pad}" y1="${height - pad}" x2="${width - pad}" y2="${height - pad}"></line>
    <line class="axis" x1="${pad}" y1="${pad}" x2="${pad}" y2="${height - pad}"></line>
    <path class="line-reference" d="M ${pad} ${height - pad} L ${width - pad} ${pad}"></path>
    ${bars}
  `;
}

function renderTable(containerId, headers, rows) {
  el(containerId).innerHTML = `
    <table>
      <thead><tr>${headers.map((header) => `<th class="${header.number ? "number" : ""}">${header.label}</th>`).join("")}</tr></thead>
      <tbody>
        ${rows.map((row) => `
          <tr>
            ${headers.map((header) => `<td class="${header.number ? "number" : ""}">${header.render(row)}</td>`).join("")}
          </tr>
        `).join("")}
      </tbody>
    </table>
  `;
}

function renderModelComparison() {
  renderTable(
    "modelComparison",
    [
      { label: "Model", render: (row) => row.name },
      { label: "Matches", number: true, render: (row) => row.matches },
      { label: "Accuracy", number: true, render: (row) => fmtNum(row.accuracy, 3) },
      { label: "Log loss", number: true, render: (row) => fmtNum(row.log_loss, 3) },
      { label: "Brier", number: true, render: (row) => fmtNum(row.brier, 3) },
    ],
    report.model_comparison || []
  );
}

function renderCompetitionTable() {
  renderTable(
    "competitionTable",
    [
      { label: "Competition", render: (row) => row.competition },
      { label: "Matches", number: true, render: (row) => row.matches },
      { label: "Accuracy", number: true, render: (row) => fmtNum(row.accuracy, 3) },
      { label: "Avg confidence", number: true, render: (row) => fmtPct(row.avg_confidence) },
    ],
    report.competition_breakdown || []
  );
}

function formStrip(form) {
  return `<span class="form-strip">${String(form || "").split("").map((value) => `<span class="${value}">${value}</span>`).join("")}</span>`;
}

function renderTeamTable() {
  renderTable(
    "teamTable",
    [
      { label: "#", number: true, render: (row) => row.rank },
      { label: "Team", render: (row) => row.team },
      { label: "Elo", number: true, render: (row) => fmtNum(row.elo, 0) },
      { label: "Power", number: true, render: (row) => fmtNum(row.power_score, 1) },
      { label: "PPM", number: true, render: (row) => fmtNum(row.points_per_match, 2) },
      { label: "xG diff", number: true, render: (row) => fmtNum(row.xg_diff_per_match, 2) },
      { label: "Rest", number: true, render: (row) => row.rest_days },
      { label: "Form", render: (row) => formStrip(row.form) },
    ],
    (report.team_profiles || []).slice(0, 24)
  );
}

function renderBettingLedger() {
  const rows = (report.betting_ledger || []).filter((row) => row.placed).slice(-24).reverse();
  renderTable(
    "bettingLedger",
    [
      { label: "Date", render: (row) => row.date },
      { label: "Match", render: (row) => row.match },
      { label: "Bet", render: (row) => row.outcome || "-" },
      { label: "Edge", number: true, render: (row) => fmtSignedPct(row.edge) },
      { label: "Odds", number: true, render: (row) => fmtNum(row.odds, 2) },
      { label: "Stake", number: true, render: (row) => fmtMoney(row.stake) },
      { label: "Profit", number: true, render: (row) => fmtMoney(row.profit) },
      { label: "Bankroll", number: true, render: (row) => fmtMoney(row.bankroll_after) },
    ],
    rows
  );
}

async function init() {
  try {
    const response = await fetch("data/report.json", { cache: "no-store" });
    report = await response.json();
  } catch (error) {
    document.body.innerHTML = `<main class="shell"><section class="panel"><h1>Report not found</h1><p class="muted">Run python -m football_predictor.cli demo --output web/data/report.json</p></section></main>`;
    return;
  }

  el("modelName").textContent = report.model_name || "model";
  el("sourceName").textContent = report.source || "source";
  renderMetricCards(report.summary || {});
  renderMatchSelector();
  renderMatch(currentMatch);
  populateLabTeams();
  renderLab();
  renderRankList("importanceList", report.feature_importance, (row) => fmtPct(row.importance));
  renderRankList("factorList", report.local_factors, (row) => row.direction_score >= 0 ? `+${fmtNum(row.direction_score, 3)}` : fmtNum(row.direction_score, 3));
  renderLineChart("bankrollChart", report.bankroll || [], "bankroll_after", "line-bankroll");
  renderCalibration();
  renderModelComparison();
  renderCompetitionTable();
  renderTeamTable();
  renderBettingLedger();
  el("notes").innerHTML = (report.notes || []).map((note) => `<li>${note}</li>`).join("");
}

init();
