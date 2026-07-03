const OUTCOMES = [
  { key: "home", label: "Home", css: "home" },
  { key: "draw", label: "Draw", css: "draw" },
  { key: "away", label: "Away", css: "away" },
];

const STORAGE_KEYS = {
  theme: "football-predictor-theme",
  watchlist: "football-predictor-watchlist-v2",
};

const state = {
  report: null,
  fixtures: [],
  selectedMatch: null,
  selectedTeam: null,
  selectedLeague: null,
  watchlist: { matches: [], teams: [] },
};

const $ = (id) => document.getElementById(id);
const clip = (value, min, max) => Math.max(min, Math.min(max, value));
const toNumber = (value, fallback = 0) => Number.isFinite(Number(value)) ? Number(value) : fallback;
const fmtPct = (value, digits = 1) => `${(toNumber(value) * 100).toFixed(digits)}%`;
const fmtSignedPct = (value, digits = 1) => `${toNumber(value) >= 0 ? "+" : ""}${fmtPct(value, digits)}`;
const fmtNum = (value, digits = 2) => toNumber(value).toFixed(digits);
const fmtInt = (value) => Math.round(toNumber(value)).toLocaleString("en-GB");
const fmtOdds = (value) => toNumber(value) > 0 ? toNumber(value).toFixed(2) : "-";
const fmtMoney = (value) => `${toNumber(value) < 0 ? "-" : ""}GBP ${Math.abs(toNumber(value)).toFixed(2)}`;

function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function initials(name) {
  const words = String(name || "?").replace(/[^A-Za-z0-9 ]/g, " ").trim().split(/\s+/).filter(Boolean);
  if (!words.length) return "?";
  if (words.length === 1) return words[0].slice(0, 3).toUpperCase();
  return words.slice(0, 2).map((word) => word[0]).join("").toUpperCase();
}

function confidenceClass(confidence) {
  return String(confidence?.label || "Low").toLowerCase();
}

function outcomeLabels(match) {
  return {
    home: match?.home_team || "Home",
    draw: "Draw",
    away: match?.away_team || "Away",
  };
}

function probabilityObject(match) {
  return {
    home: toNumber(match?.probabilities?.home ?? match?.p_home),
    draw: toNumber(match?.probabilities?.draw ?? match?.p_draw),
    away: toNumber(match?.probabilities?.away ?? match?.p_away),
  };
}

function bestProbability(match) {
  const probs = probabilityObject(match);
  return Math.max(probs.home, probs.draw, probs.away);
}

function bestEdge(match, positiveOnly = false) {
  const edges = match?.edges || [];
  if (!edges.length) return 0;
  const values = edges.map((edge) => toNumber(edge.edge));
  if (positiveOnly) return Math.max(0, ...values);
  return values.reduce((winner, value) => Math.abs(value) > Math.abs(winner) ? value : winner, 0);
}

function bestEdgeRow(match) {
  const edges = match?.edges || [];
  if (!edges.length) return null;
  return edges.reduce((winner, row) => toNumber(row.edge) > toNumber(winner.edge) ? row : winner, edges[0]);
}

function downloadText(filename, text, type = "text/plain") {
  const blob = new Blob([text], { type });
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = filename;
  document.body.appendChild(link);
  link.click();
  link.remove();
  URL.revokeObjectURL(url);
}

function csvEscape(value) {
  const text = String(value ?? "");
  return /[",\n]/.test(text) ? `"${text.replaceAll('"', '""')}"` : text;
}

function toCsv(rows, columns) {
  const header = columns.map((column) => csvEscape(column.label)).join(",");
  const body = rows.map((row) => columns.map((column) => csvEscape(column.value(row))).join(",")).join("\n");
  return `${header}\n${body}\n`;
}

function showToast(message) {
  const toast = $("toast");
  toast.textContent = message;
  toast.classList.add("show");
  clearTimeout(showToast.timer);
  showToast.timer = setTimeout(() => toast.classList.remove("show"), 1800);
}

function setOptions(select, values, selectedValue, allLabel) {
  const list = allLabel ? [{ value: "All", label: allLabel }, ...values] : values;
  select.innerHTML = list
    .map((item) => {
      const value = typeof item === "string" ? item : item.value;
      const label = typeof item === "string" ? item : item.label;
      return `<option value="${escapeHtml(value)}">${escapeHtml(label)}</option>`;
    })
    .join("");
  if (selectedValue && [...select.options].some((option) => option.value === selectedValue)) {
    select.value = selectedValue;
  }
}

function loadWatchlist() {
  try {
    const saved = JSON.parse(localStorage.getItem(STORAGE_KEYS.watchlist) || "{}");
    state.watchlist = {
      matches: Array.isArray(saved.matches) ? saved.matches : [],
      teams: Array.isArray(saved.teams) ? saved.teams : [],
    };
  } catch {
    state.watchlist = { matches: [], teams: [] };
  }
}

function saveWatchlist() {
  localStorage.setItem(STORAGE_KEYS.watchlist, JSON.stringify(state.watchlist));
}

function buildFixtures(report) {
  const seen = new Set();
  const merged = [];
  const add = (fixture, source) => {
    if (!fixture) return;
    const id = String(fixture.id || `${fixture.date}_${fixture.home_team}_${fixture.away_team}_${source}`);
    if (seen.has(id)) return;
    seen.add(id);
    merged.push({ ...fixture, id, source_type: source });
  };
  (report.forecast_fixtures || []).forEach((fixture) => add(fixture, "forecast"));
  (report.dashboard?.today || []).forEach((fixture) => add(fixture, "today"));
  (report.dashboard?.tomorrow || []).forEach((fixture) => add(fixture, "tomorrow"));
  (report.fixture_cards || []).forEach((fixture) => add(fixture, "backtest"));
  if (!merged.length && report.match_detail) add(report.match_detail, "detail");
  return merged.sort((a, b) => String(a.date || "").localeCompare(String(b.date || "")));
}

function teamProfile(name) {
  return (state.report?.team_analytics || []).find((team) => team.team === name) || null;
}

function leagueProfile(name) {
  return (state.report?.league_analytics || []).find((league) => league.competition === name) || null;
}

function setupStaticEvents() {
  $("copyLinkButton").addEventListener("click", async () => {
    try {
      await navigator.clipboard.writeText(window.location.href);
    } catch {
      const textarea = document.createElement("textarea");
      textarea.value = window.location.href;
      textarea.setAttribute("readonly", "");
      textarea.style.position = "fixed";
      textarea.style.left = "-9999px";
      document.body.appendChild(textarea);
      textarea.select();
      document.execCommand("copy");
      textarea.remove();
    }
    showToast("Link copied");
  });

  $("themeButton").addEventListener("click", () => {
    const next = document.documentElement.dataset.theme === "light" ? "dark" : "light";
    document.documentElement.dataset.theme = next;
    localStorage.setItem(STORAGE_KEYS.theme, next);
    renderCharts();
  });

  $("searchButton").addEventListener("click", openSearch);
  $("closeSearchButton").addEventListener("click", closeSearch);
  $("commandPalette").addEventListener("click", (event) => {
    if (event.target.id === "commandPalette") closeSearch();
  });
  $("commandInput").addEventListener("input", renderSearchResults);
  document.addEventListener("keydown", (event) => {
    if ((event.ctrlKey || event.metaKey) && event.key.toLowerCase() === "k") {
      event.preventDefault();
      openSearch();
    }
    if (event.key === "Escape") closeSearch();
  });

  window.addEventListener("hashchange", updateActiveNav);
}

function setupDataEvents() {
  ["dashboardLeagueFilter", "fixtureLeagueFilter", "fixtureSort", "confidenceFilter", "edgeFilter", "fixtureSearch", "oddsOnlyFilter"]
    .forEach((id) => $(id).addEventListener("input", () => {
      renderDashboard();
      renderFixtures();
    }));

  $("matchSelect").addEventListener("change", () => {
    const match = state.fixtures.find((fixture) => fixture.id === $("matchSelect").value);
    if (match) setSelectedMatch(match, false);
  });

  $("saveMatchButton").addEventListener("click", () => {
    if (!state.selectedMatch) return;
    if (!state.watchlist.matches.includes(state.selectedMatch.id)) {
      state.watchlist.matches.unshift(state.selectedMatch.id);
      state.watchlist.matches = state.watchlist.matches.slice(0, 20);
      saveWatchlist();
      renderWatchlist();
      showToast("Match saved");
    }
  });

  $("teamSelect").addEventListener("change", () => {
    state.selectedTeam = $("teamSelect").value;
    renderTeams();
  });

  $("teamSearch").addEventListener("input", renderTeams);

  $("saveTeamButton").addEventListener("click", () => {
    if (!state.selectedTeam) return;
    if (!state.watchlist.teams.includes(state.selectedTeam)) {
      state.watchlist.teams.unshift(state.selectedTeam);
      state.watchlist.teams = state.watchlist.teams.slice(0, 20);
      saveWatchlist();
      renderWatchlist();
      showToast("Team saved");
    }
  });

  $("leagueSelect").addEventListener("change", () => {
    state.selectedLeague = $("leagueSelect").value;
    renderLeagues();
  });

  $("backtestLeagueFilter").addEventListener("change", renderBacktest);

  $("exportPredictionsButton").addEventListener("click", exportPredictionsCsv);
  $("exportBacktestButton").addEventListener("click", exportBacktestCsv);
  $("downloadReportButton").addEventListener("click", () => {
    downloadText("football-predictor-report.json", JSON.stringify(state.report, null, 2), "application/json");
  });

  $("fixturesGrid").addEventListener("click", (event) => {
    const button = event.target.closest("[data-match-id]");
    if (!button) return;
    const match = state.fixtures.find((fixture) => fixture.id === button.dataset.matchId);
    if (match) {
      setSelectedMatch(match, true);
    }
  });

  $("featuredFixtures").addEventListener("click", (event) => {
    const button = event.target.closest("[data-match-id]");
    if (!button) return;
    const match = state.fixtures.find((fixture) => fixture.id === button.dataset.matchId);
    if (match) setSelectedMatch(match, true);
  });

  ["labHomeTeam", "labAwayTeam", "labNeutral", "labHomeOdds", "labDrawOdds", "labAwayOdds", "labBankroll"]
    .forEach((id) => $(id).addEventListener("input", renderLabSimulation));

  $("swapTeamsButton").addEventListener("click", () => {
    const home = $("labHomeTeam").value;
    $("labHomeTeam").value = $("labAwayTeam").value;
    $("labAwayTeam").value = home;
    renderLabSimulation();
  });

  $("randomFixtureButton").addEventListener("click", () => {
    const teams = state.report.team_analytics || [];
    if (teams.length < 2) return;
    const a = Math.floor(Math.random() * teams.length);
    let b = Math.floor(Math.random() * teams.length);
    if (b === a) b = (b + 1) % teams.length;
    $("labHomeTeam").value = teams[a].team;
    $("labAwayTeam").value = teams[b].team;
    renderLabSimulation();
  });

  $("runBacktestButton").addEventListener("click", () => showToast("Backtest report is generated by the Python pipeline"));
  $("retrainButton").addEventListener("click", () => showToast("Retraining hook is ready for a backend worker"));
  $("clearWatchlistButton").addEventListener("click", () => {
    state.watchlist = { matches: [], teams: [] };
    saveWatchlist();
    renderWatchlist();
    showToast("Watchlist cleared");
  });
}

function populateControls() {
  const leagues = [...new Set(state.fixtures.map((fixture) => fixture.competition).filter(Boolean))].sort();
  setOptions($("dashboardLeagueFilter"), leagues, "All", "All leagues");
  setOptions($("fixtureLeagueFilter"), leagues, "All", "All leagues");
  setOptions($("backtestLeagueFilter"), leagues, "All", "All leagues");

  $("matchSelect").innerHTML = state.fixtures
    .map((fixture) => `<option value="${escapeHtml(fixture.id)}">${escapeHtml(`${fixture.date} - ${fixture.home_team} vs ${fixture.away_team}`)}</option>`)
    .join("");

  const teams = (state.report.team_analytics || []).map((team) => team.team);
  setOptions($("teamSelect"), teams, teams[0]);
  setOptions($("labHomeTeam"), teams, teams[0]);
  setOptions($("labAwayTeam"), teams, teams[1] || teams[0]);

  const analyticsLeagues = (state.report.league_analytics || []).map((league) => league.competition);
  setOptions($("leagueSelect"), analyticsLeagues, analyticsLeagues[0]);

  state.selectedTeam = $("teamSelect").value;
  state.selectedLeague = $("leagueSelect").value;
}

function setSelectedMatch(match, navigate) {
  state.selectedMatch = match;
  if ($("matchSelect").value !== match.id) $("matchSelect").value = match.id;
  renderMatch();
  if (navigate) {
    window.location.hash = "match";
    $("match").scrollIntoView({ behavior: "smooth", block: "start" });
  }
}

function renderStatus() {
  const report = state.report;
  $("sideModelName").textContent = report.model_name || "Model";
  $("sideSourceName").textContent = report.source || "Local report";
  $("dashboardChips").innerHTML = [
    `<span class="chip good">${escapeHtml(report.generated_at || "Generated")}</span>`,
    `<span class="chip">${fmtInt(report.data_quality?.scored_matches)} scored matches</span>`,
    `<span class="chip warn">xG coverage ${fmtPct(report.data_quality?.xg_coverage || 0, 0)}</span>`,
  ].join("");
}

function renderMetricCards(containerId, cards) {
  $(containerId).innerHTML = cards.map((card) => `
    <div class="metric-card">
      <span>${escapeHtml(card.label)}</span>
      <strong>${escapeHtml(card.value)}</strong>
      <div class="delta">${escapeHtml(card.detail || "")}</div>
    </div>
  `).join("");
}

function renderDashboard() {
  const summary = state.report.summary || {};
  const betting = summary.betting || {};
  renderMetricCards("summaryCards", [
    { label: "Backtest matches", value: fmtInt(summary.matches), detail: "Chronological walk-forward" },
    { label: "Accuracy", value: fmtPct(summary.accuracy), detail: "1X2 direction" },
    { label: "Log loss", value: fmtNum(summary.log_loss, 3), detail: "Lower is better" },
    { label: "Brier score", value: fmtNum(summary.brier, 3), detail: "Probability error" },
    { label: "Betting ROI", value: fmtSignedPct(betting.roi), detail: `${fmtInt(betting.bets)} simulated bets` },
    { label: "Max drawdown", value: fmtSignedPct(betting.max_drawdown), detail: "Historical simulation" },
  ]);

  const selectedLeague = $("dashboardLeagueFilter").value || "All";
  const candidates = [
    ...(state.report.dashboard?.today || []),
    ...(state.report.dashboard?.tomorrow || []),
    ...(state.report.forecast_fixtures || []),
    ...(state.report.dashboard?.top_picks || []),
  ];
  const featured = candidates
    .filter((fixture, index, list) => list.findIndex((item) => item.id === fixture.id) === index)
    .filter((fixture) => selectedLeague === "All" || fixture.competition === selectedLeague)
    .sort((a, b) => bestProbability(b) - bestProbability(a))
    .slice(0, 6);
  $("featuredFixtures").innerHTML = featured.length ? featured.map(renderFixtureCard).join("") : emptyState("No fixtures match the selected league.");

  const disagreements = (state.report.dashboard?.market_disagreements || state.fixtures)
    .filter((fixture) => selectedLeague === "All" || fixture.competition === selectedLeague)
    .sort((a, b) => Math.abs(bestEdge(b)) - Math.abs(bestEdge(a)))
    .slice(0, 8);
  $("marketDisagreements").innerHTML = disagreements.map((fixture) => {
    const edge = bestEdgeRow(fixture);
    return `
      <button class="stack-item" data-match-id="${escapeHtml(fixture.id)}" type="button">
        <span>
          <strong>${escapeHtml(fixture.home_team)} vs ${escapeHtml(fixture.away_team)}</strong>
          <small>${escapeHtml(fixture.competition || "")}</small>
        </span>
        <span class="pill ${toNumber(edge?.edge) >= 0 ? "good" : "bad"}">${escapeHtml(edge?.label || "Edge")} ${fmtSignedPct(edge?.edge)}</span>
      </button>
    `;
  }).join("");
  $("marketDisagreements").querySelectorAll("[data-match-id]").forEach((button) => {
    button.addEventListener("click", () => {
      const match = state.fixtures.find((fixture) => fixture.id === button.dataset.matchId);
      if (match) setSelectedMatch(match, true);
    });
  });

  drawCalibration("dashboardCalibration", state.report.calibration || []);
  drawLineChart("accuracyTrend", state.report.backtest_dashboard?.rolling || [], [
    { key: "accuracy", label: "Accuracy", className: "line-blue" },
    { key: "log_loss", label: "Log loss", className: "line-amber", invert: true },
  ], { yMin: 0.3, yMax: 1.1 });
}

function fixtureFilterPredicate(fixture) {
  const search = $("fixtureSearch").value.trim().toLowerCase();
  const league = $("fixtureLeagueFilter").value || "All";
  const minConfidence = toNumber($("confidenceFilter").value) / 100;
  const minEdge = toNumber($("edgeFilter").value) / 100;
  const oddsOnly = $("oddsOnlyFilter").checked;
  const haystack = `${fixture.home_team} ${fixture.away_team} ${fixture.competition}`.toLowerCase();
  const hasOdds = (fixture.edges || []).some((edge) => toNumber(edge.bookmaker_odds) > 1);
  return (!search || haystack.includes(search))
    && (league === "All" || fixture.competition === league)
    && toNumber(fixture.confidence?.score) / 100 >= minConfidence
    && Math.max(0, bestEdge(fixture, true)) >= minEdge
    && (!oddsOnly || hasOdds);
}

function renderFixtures() {
  const sort = $("fixtureSort").value;
  const rows = state.fixtures.filter(fixtureFilterPredicate);
  rows.sort((a, b) => {
    if (sort === "confidence") return toNumber(b.confidence?.score) - toNumber(a.confidence?.score);
    if (sort === "edge") return bestEdge(b, true) - bestEdge(a, true);
    if (sort === "xg") return toNumber(b.expected_goals?.total) - toNumber(a.expected_goals?.total);
    if (sort === "upset") return Math.max(toNumber(b.probabilities?.away), toNumber(b.probabilities?.draw)) - Math.max(toNumber(a.probabilities?.away), toNumber(a.probabilities?.draw));
    return String(a.date || "").localeCompare(String(b.date || ""));
  });
  $("fixturesGrid").innerHTML = rows.length ? rows.slice(0, 72).map(renderFixtureCard).join("") : emptyState("No fixtures match the current filters.");
}

function renderFixtureCard(match) {
  const probs = probabilityObject(match);
  const labels = outcomeLabels(match);
  const edge = bestEdgeRow(match);
  const confidence = match.confidence || {};
  const selected = state.selectedMatch?.id === match.id ? " selected" : "";
  return `
    <article class="fixture-card${selected}" data-match-id="${escapeHtml(match.id)}">
      <div class="fixture-meta">
        <span>${escapeHtml(match.date || "-")}</span>
        <span>${escapeHtml(match.competition || "-")}</span>
      </div>
      <div class="team-row">
        <span class="badge">${escapeHtml(initials(match.home_team))}</span>
        <strong>${escapeHtml(match.home_team)}</strong>
        <span>${fmtPct(probs.home, 0)}</span>
      </div>
      <div class="team-row">
        <span class="badge">${escapeHtml(initials(match.away_team))}</span>
        <strong>${escapeHtml(match.away_team)}</strong>
        <span>${fmtPct(probs.away, 0)}</span>
      </div>
      <div class="mini-bars">
        ${OUTCOMES.map((outcome) => `<div class="mini-bar" title="${escapeHtml(labels[outcome.key])} ${fmtPct(probs[outcome.key])}"><span style="width:${clip(probs[outcome.key] * 100, 2, 100)}%"></span></div>`).join("")}
      </div>
      <div class="card-footer">
        <span class="prediction-score">${escapeHtml(match.predicted_score || "-")}</span>
        <span class="confidence-chip ${confidenceClass(confidence)}">${escapeHtml(confidence.label || "Low")} ${fmtInt(confidence.score || 0)}</span>
      </div>
      <div class="card-footer">
        <small>xG ${fmtNum(match.expected_goals?.home)} - ${fmtNum(match.expected_goals?.away)}</small>
        <span class="pill ${toNumber(edge?.edge) > 0 ? "good" : toNumber(edge?.edge) < 0 ? "bad" : ""}">${fmtSignedPct(edge?.edge || 0)}</span>
      </div>
      <button data-match-id="${escapeHtml(match.id)}" type="button">View Prediction</button>
    </article>
  `;
}

function renderMatch() {
  const match = state.selectedMatch || state.report.match_detail || state.fixtures[0];
  if (!match) return;
  state.selectedMatch = match;
  $("matchTitle").textContent = `${match.home_team} vs ${match.away_team}`;
  $("matchSelect").value = match.id;
  renderMatchHeader(match);
  renderOutcomeBars("matchProbabilityBars", probabilityObject(match), outcomeLabels(match));
  renderMarketComparison(match);
  renderGoalMarkets(match);
  renderHeatmap("scoreHeatmap", match.score_matrix || state.report.score_matrix);
  renderFeatureDrivers(match);
  drawRadar("teamRadar", match.home_team, match.away_team);
  renderSimilarMatches(match);
}

function renderMatchHeader(match) {
  const confidence = match.confidence || {};
  const actual = match.actual ? `${match.actual.home_goals}-${match.actual.away_goals}` : "Pending";
  $("matchHeader").innerHTML = `
    <div class="hero-team">
      <span class="badge">${escapeHtml(initials(match.home_team))}</span>
      <strong>${escapeHtml(match.home_team)}</strong>
      <small>${escapeHtml(match.competition || "")}</small>
    </div>
    <div class="hero-center">
      <span class="prediction-score">${escapeHtml(match.predicted_score || "-")}</span>
      <span class="confidence-chip ${confidenceClass(confidence)}">${escapeHtml(confidence.label || "Low")} confidence</span>
      <small>${escapeHtml(match.date || "")} - actual ${escapeHtml(actual)}</small>
      <div class="xg-line"><span>xG ${fmtNum(match.expected_goals?.home)}</span><span>total ${fmtNum(match.expected_goals?.total)}</span><span>${fmtNum(match.expected_goals?.away)}</span></div>
    </div>
    <div class="hero-team away">
      <span class="badge">${escapeHtml(initials(match.away_team))}</span>
      <strong>${escapeHtml(match.away_team)}</strong>
      <small>Model ${escapeHtml(match.model_version || state.report.model_name || "")}</small>
    </div>
  `;
}

function renderOutcomeBars(containerId, probabilities, labels) {
  $(containerId).innerHTML = OUTCOMES.map((outcome) => {
    const value = clip(toNumber(probabilities[outcome.key]), 0, 1);
    return `
      <div class="prob-row">
        <span>${escapeHtml(labels[outcome.key] || outcome.label)}</span>
        <div class="bar-track"><div class="bar-fill ${outcome.css}" style="width:${Math.max(2, value * 100)}%"></div></div>
        <strong>${fmtPct(value)}</strong>
      </div>
    `;
  }).join("");
}

function renderMarketComparison(match) {
  const rows = match.edges || [];
  $("marketComparison").innerHTML = rows.length ? rows.map((row) => `
    <div class="value-row">
      <span>
        <strong>${escapeHtml(row.label)}</strong>
        <small>model ${fmtPct(row.model_probability)} / market ${fmtPct(row.market_probability)}</small>
      </span>
      <span class="pill ${toNumber(row.edge) > 0 ? "good" : toNumber(row.edge) < 0 ? "bad" : ""}">
        ${fmtSignedPct(row.edge)} - fair ${fmtOdds(row.fair_odds)} / book ${fmtOdds(row.bookmaker_odds)}
      </span>
    </div>
  `).join("") : emptyState("No market odds available for this match.");
}

function renderGoalMarkets(match) {
  const markets = match.markets || {};
  const rows = [
    ["Over 1.5", markets.over_1_5],
    ["Over 2.5", markets.over_2_5],
    ["Over 3.5", markets.over_3_5],
    ["BTTS", markets.btts],
    [`${match.home_team} clean sheet`, markets.home_clean_sheet],
    [`${match.away_team} clean sheet`, markets.away_clean_sheet],
  ];
  $("goalMarkets").innerHTML = rows.map(([label, value]) => `
    <div class="stat-row">
      <span>${escapeHtml(label)}</span>
      <strong>${fmtPct(value)}</strong>
    </div>
  `).join("");
}

function renderHeatmap(containerId, matrix) {
  const safe = Array.isArray(matrix) ? matrix : [];
  const flat = safe.flat().map(toNumber);
  const max = Math.max(...flat, 0.0001);
  let html = "";
  safe.forEach((row, homeGoals) => {
    row.forEach((value, awayGoals) => {
      const ratio = clip(toNumber(value) / max, 0, 1);
      const alpha = 0.16 + (0.82 * ratio);
      const color = `rgba(53, 167, 255, ${alpha})`;
      html += `<div class="heat-cell" style="background:${color}" title="${homeGoals}-${awayGoals}: ${fmtPct(value)}">${homeGoals}-${awayGoals}<br>${fmtPct(value, 0)}</div>`;
    });
  });
  $(containerId).innerHTML = html;
}

function renderFeatureDrivers(match) {
  const edge = bestEdgeRow(match);
  const drivers = [
    { label: "Team strength", detail: `${match.home_team} and ${match.away_team} Elo/form priors`, value: bestProbability(match) - 0.333 },
    { label: "Expected goals", detail: `${fmtNum(match.expected_goals?.home)} to ${fmtNum(match.expected_goals?.away)}`, value: toNumber(match.expected_goals?.home) - toNumber(match.expected_goals?.away) },
    { label: "Market disagreement", detail: `${edge?.label || "Top edge"} ${fmtSignedPct(edge?.edge)}`, value: toNumber(edge?.edge) },
    { label: "Draw pressure", detail: `Draw probability ${fmtPct(match.probabilities?.draw)}`, value: toNumber(match.probabilities?.draw) - 0.25 },
    { label: "Data quality", detail: state.report.data_quality?.warnings?.[0] || "Core result and odds feeds available", value: toNumber(state.report.data_quality?.odds_coverage) - 0.5 },
  ];
  const importance = (state.report.feature_importance || []).slice(0, 5).map((row) => ({
    label: row.label || row.feature,
    detail: "Global model feature importance",
    value: toNumber(row.importance),
  }));
  $("featureDrivers").innerHTML = [
    `<div class="value-row"><span><strong>Plain English</strong><small>${escapeHtml(match.explanation || "")}</small></span></div>`,
    ...drivers.map(renderRankRow),
    ...importance.map(renderRankRow),
    ...(match.risk_factors || []).slice(0, 3).map((risk) => `<div class="value-row"><span><strong>Risk</strong><small>${escapeHtml(risk)}</small></span></div>`),
  ].join("");
}

function renderRankRow(row) {
  const magnitude = clip(Math.abs(toNumber(row.value)) * 100, 3, 100);
  return `
    <div class="rank-row">
      <div class="rank-row-head">
        <strong>${escapeHtml(row.label)}</strong>
        <span>${toNumber(row.value) >= 0 ? "+" : ""}${fmtNum(row.value, 3)}</span>
      </div>
      <div class="rank-track"><span class="rank-fill ${toNumber(row.value) < 0 ? "negative" : ""}" style="width:${magnitude}%"></span></div>
      <small>${escapeHtml(row.detail || "")}</small>
    </div>
  `;
}

function renderSimilarMatches(match) {
  const rows = match.similar_matches || state.report.match_detail?.similar_matches || [];
  $("similarMatches").innerHTML = rows.length ? rows.slice(0, 8).map((row) => `
    <div class="stack-item">
      <span>
        <strong>${escapeHtml(row.home_team)} vs ${escapeHtml(row.away_team)}</strong>
        <small>${escapeHtml(row.date || "")} - ${escapeHtml(row.competition || "")}</small>
      </span>
      <span class="pill">${escapeHtml(row.score || `${row.home_goals ?? "-"}-${row.away_goals ?? "-"}`)} - sim ${fmtPct(row.similarity || 0, 0)}</span>
    </div>
  `).join("") : emptyState("Similar-match archive is not available for this fixture.");
}

function renderTeams() {
  const selected = teamProfile(state.selectedTeam) || (state.report.team_analytics || [])[0];
  if (!selected) return;
  $("teamSelect").value = selected.team;
  renderTeamProfile(selected);
  drawTeamTrend(selected);
  const search = $("teamSearch").value.trim().toLowerCase();
  const rows = (state.report.team_analytics || [])
    .filter((team) => !search || team.team.toLowerCase().includes(search))
    .slice(0, 40);
  renderTable("teamTable", rows, [
    { label: "Rank", value: (row) => row.rank },
    { label: "Team", value: (row) => row.team },
    { label: "Elo", value: (row) => fmtInt(row.elo) },
    { label: "PPG", value: (row) => fmtNum(row.points_per_match, 2) },
    { label: "Win %", value: (row) => fmtPct(row.win_rate, 0) },
    { label: "GF", value: (row) => fmtNum(row.goals_for_per_match, 2) },
    { label: "GA", value: (row) => fmtNum(row.goals_against_per_match, 2) },
    { label: "Form", value: (row) => row.form },
  ]);
}

function renderTeamProfile(team) {
  $("teamProfile").innerHTML = `
    <div class="profile-title">
      <span class="badge">${escapeHtml(initials(team.team))}</span>
      <span>
        <strong>${escapeHtml(team.team)}</strong>
        <small>Rank ${fmtInt(team.rank)} - Elo ${fmtInt(team.elo)}</small>
      </span>
    </div>
    <div class="form-string">${String(team.form || "").split("").map((result) => `<span class="form-pill ${result.toLowerCase()}">${escapeHtml(result)}</span>`).join("")}</div>
    <div class="stat-stack">
      ${[
        ["Matches", fmtInt(team.matches)],
        ["Points per match", fmtNum(team.points_per_match, 2)],
        ["Goal diff / match", fmtNum(team.goal_diff_per_match, 2)],
        ["xG diff / match", fmtNum(team.xg_diff_per_match, 2)],
        ["Recent points", fmtNum(team.recent_points, 2)],
        ["Availability", team.availability || "No feed"],
      ].map(([label, value]) => `<div class="stat-row"><span>${escapeHtml(label)}</span><strong>${escapeHtml(value)}</strong></div>`).join("")}
    </div>
  `;
}

function renderLeagues() {
  const league = leagueProfile(state.selectedLeague) || (state.report.league_analytics || [])[0];
  if (!league) return;
  $("leagueSelect").value = league.competition;
  $("leagueSummary").innerHTML = `
    <div class="profile-title">
      <span class="badge">LG</span>
      <span>
        <strong>${escapeHtml(league.competition)}</strong>
        <small>${fmtInt(league.teams)} teams - ${fmtInt(league.matches)} matches</small>
      </span>
    </div>
    <div class="stat-stack">
      <div class="stat-row"><span>Average goals</span><strong>${fmtNum(league.avg_goals, 2)}</strong></div>
      <div class="stat-row"><span>Home win rate</span><strong>${fmtPct(league.home_win_rate)}</strong></div>
      <div class="stat-row"><span>Draw rate</span><strong>${fmtPct(league.draw_rate)}</strong></div>
    </div>
  `;
  $("leaguePerf").innerHTML = `
    <div class="stat-row"><span>Accuracy</span><strong>${fmtPct(league.accuracy)}</strong></div>
    <div class="stat-row"><span>Log loss</span><strong>${fmtNum(league.log_loss, 3)}</strong></div>
    <div class="stat-row"><span>Predictability</span><strong>${fmtPct(league.accuracy || 0)}</strong></div>
  `;
  renderTable("leagueTable", league.standings || [], [
    { label: "#", value: (_row, index) => index + 1 },
    { label: "Team", value: (row) => row.team },
    { label: "Matches", value: (row) => fmtInt(row.matches) },
    { label: "Points", value: (row) => fmtInt(row.points) },
    { label: "Goal diff", value: (row) => fmtInt(row.goal_diff) },
    { label: "Power", value: (row) => fmtNum(row.power_score, 1) },
  ]);
  drawLeagueQuadrant(league);
}

function renderBacktest() {
  const selectedLeague = $("backtestLeagueFilter").value || "All";
  const summary = state.report.summary || {};
  const betting = summary.betting || {};
  const comp = (state.report.competition_breakdown || []).find((row) => row.competition === selectedLeague);
  renderMetricCards("backtestCards", [
    { label: "Scope", value: selectedLeague === "All" ? "All leagues" : selectedLeague, detail: "Walk-forward holdout" },
    { label: "Matches", value: fmtInt(comp?.matches ?? summary.matches), detail: "Predicted before result update" },
    { label: "Accuracy", value: fmtPct(comp?.accuracy ?? summary.accuracy), detail: "Directional result" },
    { label: "Avg confidence", value: fmtPct(comp?.avg_confidence ?? bestProbability(state.selectedMatch)), detail: "Selected scope" },
    { label: "ROI sim", value: fmtSignedPct(betting.roi), detail: "Analytical, not advice" },
    { label: "Ending bankroll", value: fmtMoney(betting.ending_bankroll), detail: `Max DD ${fmtSignedPct(betting.max_drawdown)}` },
  ]);

  $("confidenceBuckets").innerHTML = (state.report.backtest_dashboard?.confidence_buckets || []).map((row) => renderRankRow({
    label: row.bucket,
    detail: `${fmtInt(row.count)} matches - mean confidence ${fmtPct(row.mean_confidence)}`,
    value: toNumber(row.accuracy),
  })).join("");

  renderConfusionMatrix(state.report.backtest_dashboard?.confusion_matrix || []);
  drawCalibration("calibrationChart", state.report.calibration || []);
  drawBankroll("bankrollChart", state.report.bankroll || []);

  const archive = (state.report.backtest_dashboard?.archive || [])
    .filter((row) => selectedLeague === "All" || row.competition === selectedLeague)
    .slice(0, 28);
  renderTable("predictionArchive", archive, [
    { label: "Date", value: (row) => row.date },
    { label: "Match", value: (row) => `${row.home_team} vs ${row.away_team}` },
    { label: "Pred", value: (row) => row.top_score || "-" },
    { label: "H", value: (row) => fmtPct(row.p_home, 0) },
    { label: "D", value: (row) => fmtPct(row.p_draw, 0) },
    { label: "A", value: (row) => fmtPct(row.p_away, 0) },
    { label: "Actual", value: (row) => `${fmtInt(row.home_goals)}-${fmtInt(row.away_goals)}` },
  ]);
}

function renderConfusionMatrix(matrix) {
  const labels = ["Home", "Draw", "Away"];
  const max = Math.max(...matrix.flat().map(toNumber), 1);
  let html = `<div class="confusion-cell header"></div>${labels.map((label) => `<div class="confusion-cell header">Pred ${label}</div>`).join("")}`;
  matrix.forEach((row, rowIndex) => {
    html += `<div class="confusion-cell header">Actual ${labels[rowIndex]}</div>`;
    row.forEach((value) => {
      const alpha = 0.08 + 0.7 * (toNumber(value) / max);
      html += `<div class="confusion-cell" style="background:rgba(53,167,255,${alpha})">${fmtInt(value)}</div>`;
    });
  });
  $("confusionMatrix").innerHTML = html;
}

function renderLab() {
  $("modelRegistry").innerHTML = (state.report.model_registry || []).map((model) => `
    <div class="stack-item">
      <span>
        <strong>${escapeHtml(model.name)}</strong>
        <small>${escapeHtml(model.role)}</small>
      </span>
      <span class="pill">${escapeHtml(model.version || model.status || "")}</span>
    </div>
  `).join("");

  $("leakageAudit").innerHTML = (state.report.leakage_audit || []).map((item) => `
    <div class="stack-item">
      <span>
        <strong>${escapeHtml(item.rule)}</strong>
        <small>${escapeHtml(item.detail)}</small>
      </span>
      <span class="status-chip ${escapeHtml(String(item.status || "").toLowerCase())}">${escapeHtml(item.status || "")}</span>
    </div>
  `).join("");

  renderTable("modelComparison", state.report.model_comparison || [], [
    { label: "Model", value: (row) => row.name },
    { label: "Kind", value: (row) => row.kind },
    { label: "Matches", value: (row) => fmtInt(row.matches) },
    { label: "Accuracy", value: (row) => fmtPct(row.accuracy) },
    { label: "Log loss", value: (row) => fmtNum(row.log_loss, 3) },
    { label: "Brier", value: (row) => fmtNum(row.brier, 3) },
  ]);

  renderDataQuality();
  renderWatchlist();
  renderLabSimulation();
}

function renderDataQuality() {
  const quality = state.report.data_quality || {};
  const rows = [
    ["Matches", fmtInt(quality.matches)],
    ["Scored matches", fmtInt(quality.scored_matches)],
    ["Teams", fmtInt(quality.teams)],
    ["Competitions", fmtInt(quality.competitions)],
    ["Date range", `${quality.date_start || "-"} to ${quality.date_end || "-"}`],
    ["Odds coverage", fmtPct(quality.odds_coverage)],
    ["xG coverage", fmtPct(quality.xg_coverage)],
    ["Duplicates", fmtInt(quality.duplicate_matches)],
  ];
  $("dataQualityPanel").innerHTML = [
    ...rows.map(([label, value]) => `<div class="quality-row"><span>${escapeHtml(label)}</span><strong>${escapeHtml(value)}</strong></div>`),
    ...(quality.warnings || []).map((warning) => `<div class="value-row"><span><strong>Warning</strong><small>${escapeHtml(warning)}</small></span></div>`),
    ...(state.report.responsible_use || []).map((line) => `<div class="value-row"><span><strong>Responsible use</strong><small>${escapeHtml(line)}</small></span></div>`),
  ].join("");
}

function renderWatchlist() {
  const matches = state.watchlist.matches
    .map((id) => state.fixtures.find((fixture) => fixture.id === id))
    .filter(Boolean);
  const teams = state.watchlist.teams
    .map((name) => teamProfile(name))
    .filter(Boolean);
  const html = [
    ...matches.map((match) => `
      <button class="stack-item" data-watch-match="${escapeHtml(match.id)}" type="button">
        <span><strong>${escapeHtml(match.home_team)} vs ${escapeHtml(match.away_team)}</strong><small>${escapeHtml(match.date || "")}</small></span>
        <span class="pill">${fmtPct(bestProbability(match), 0)}</span>
      </button>
    `),
    ...teams.map((team) => `
      <button class="stack-item" data-watch-team="${escapeHtml(team.team)}" type="button">
        <span><strong>${escapeHtml(team.team)}</strong><small>Elo ${fmtInt(team.elo)}</small></span>
        <span class="pill">${escapeHtml(team.form || "")}</span>
      </button>
    `),
  ].join("");
  $("watchlistPanel").innerHTML = html || emptyState("No saved matches or teams.");
  $("watchlistPanel").querySelectorAll("[data-watch-match]").forEach((button) => {
    button.addEventListener("click", () => {
      const match = state.fixtures.find((fixture) => fixture.id === button.dataset.watchMatch);
      if (match) setSelectedMatch(match, true);
    });
  });
  $("watchlistPanel").querySelectorAll("[data-watch-team]").forEach((button) => {
    button.addEventListener("click", () => {
      state.selectedTeam = button.dataset.watchTeam;
      $("teamSelect").value = state.selectedTeam;
      renderTeams();
      window.location.hash = "teams";
      $("teams").scrollIntoView({ behavior: "smooth", block: "start" });
    });
  });
}

function renderLabSimulation() {
  const home = teamProfile($("labHomeTeam").value);
  const away = teamProfile($("labAwayTeam").value);
  if (!home || !away) return;
  const neutral = $("labNeutral").checked;
  const simulated = simulateFixture(home, away, neutral);
  renderOutcomeBars("labProbabilityBars", simulated.probabilities, {
    home: home.team,
    draw: "Draw",
    away: away.team,
  });
  renderValueTable("labValueTable", simulated.probabilities, [
    toNumber($("labHomeOdds").value, 2.4),
    toNumber($("labDrawOdds").value, 3.3),
    toNumber($("labAwayOdds").value, 2.9),
  ], toNumber($("labBankroll").value, 1000));
}

function simulateFixture(home, away, neutral) {
  const eloDiff = toNumber(home.elo) - toNumber(away.elo) + (neutral ? 0 : 65);
  const homeXg = clip(1.18 + (eloDiff / 620) + (toNumber(home.attack_form) - toNumber(away.defense_form)) * 0.16 + (neutral ? 0 : 0.18), 0.25, 4.8);
  const awayXg = clip(1.12 - (eloDiff / 720) + (toNumber(away.attack_form) - toNumber(home.defense_form)) * 0.16, 0.25, 4.8);
  const matrix = scoreMatrix(homeXg, awayXg);
  const probabilities = matrixOutcomeProbabilities(matrix);
  return { homeXg, awayXg, matrix, probabilities };
}

function poissonPmf(k, lambda) {
  let factorial = 1;
  for (let i = 2; i <= k; i += 1) factorial *= i;
  return Math.exp(-lambda) * (lambda ** k) / factorial;
}

function scoreMatrix(homeXg, awayXg) {
  const rho = -0.07;
  const matrix = [];
  for (let h = 0; h <= 7; h += 1) {
    const row = [];
    for (let a = 0; a <= 7; a += 1) {
      let value = poissonPmf(h, homeXg) * poissonPmf(a, awayXg);
      if (h === 0 && a === 0) value *= Math.max(0.05, 1 - homeXg * awayXg * rho);
      if (h === 0 && a === 1) value *= Math.max(0.05, 1 + homeXg * rho);
      if (h === 1 && a === 0) value *= Math.max(0.05, 1 + awayXg * rho);
      if (h === 1 && a === 1) value *= Math.max(0.05, 1 - rho);
      row.push(value);
    }
    matrix.push(row);
  }
  const total = matrix.flat().reduce((sum, value) => sum + value, 0) || 1;
  return matrix.map((row) => row.map((value) => value / total));
}

function matrixOutcomeProbabilities(matrix) {
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
  const total = home + draw + away || 1;
  return { home: home / total, draw: draw / total, away: away / total };
}

function noVig(odds) {
  if (odds.some((value) => toNumber(value) <= 1)) return null;
  const implied = odds.map((value) => 1 / toNumber(value));
  const total = implied.reduce((sum, value) => sum + value, 0);
  return implied.map((value) => value / total);
}

function kellyFraction(probability, odds) {
  const b = toNumber(odds) - 1;
  if (b <= 0) return 0;
  return Math.max(0, ((b * probability) - (1 - probability)) / b);
}

function renderValueTable(containerId, probabilities, odds, bankroll) {
  const market = noVig(odds);
  if (!market) {
    $(containerId).innerHTML = emptyState("Enter valid decimal odds.");
    return;
  }
  const maxStake = toNumber(state.report.betting_config?.max_stake_fraction, 0.02);
  const fractionalKelly = toNumber(state.report.betting_config?.fractional_kelly, 0.2);
  $(containerId).innerHTML = OUTCOMES.map((outcome, index) => {
    const model = toNumber(probabilities[outcome.key]);
    const edge = model - market[index];
    const ev = model * odds[index] - 1;
    const stake = bankroll * Math.min(maxStake, fractionalKelly * kellyFraction(model, odds[index]));
    return `
      <div class="value-row">
        <span>
          <strong>${escapeHtml(outcome.label)}</strong>
          <small>market ${fmtPct(market[index])} / EV ${fmtSignedPct(ev)}</small>
        </span>
        <span class="pill ${edge > 0.03 ? "good" : edge < -0.03 ? "bad" : ""}">${fmtSignedPct(edge)} - ${fmtMoney(stake)}</span>
      </div>
    `;
  }).join("");
}

function renderTable(containerId, rows, columns) {
  if (!rows.length) {
    $(containerId).innerHTML = emptyState("No rows available.");
    return;
  }
  $(containerId).innerHTML = `
    <table>
      <thead><tr>${columns.map((column) => `<th>${escapeHtml(column.label)}</th>`).join("")}</tr></thead>
      <tbody>
        ${rows.map((row, index) => `
          <tr>${columns.map((column) => `<td>${escapeHtml(column.value(row, index))}</td>`).join("")}</tr>
        `).join("")}
      </tbody>
    </table>
  `;
}

function emptyState(text) {
  return `<div class="empty-state">${escapeHtml(text)}</div>`;
}

function drawLineChart(svgId, rows, series, options = {}) {
  const svg = $(svgId);
  if (!svg) return;
  const width = 840;
  const height = 280;
  const pad = { left: 44, right: 20, top: 20, bottom: 34 };
  const data = rows || [];
  if (!data.length) {
    svg.innerHTML = "";
    return;
  }
  const values = series.flatMap((item) => data.map((row) => toNumber(row[item.key])));
  const min = options.yMin ?? Math.min(...values);
  const max = options.yMax ?? Math.max(...values);
  const yScale = (value) => pad.top + (1 - ((value - min) / ((max - min) || 1))) * (height - pad.top - pad.bottom);
  const xScale = (index) => pad.left + (index / Math.max(1, data.length - 1)) * (width - pad.left - pad.right);
  const grid = [0, 0.25, 0.5, 0.75, 1].map((ratio) => {
    const y = pad.top + ratio * (height - pad.top - pad.bottom);
    const label = max - ratio * (max - min);
    return `<line class="grid" x1="${pad.left}" y1="${y}" x2="${width - pad.right}" y2="${y}"></line><text x="6" y="${y + 4}">${fmtNum(label, 2)}</text>`;
  }).join("");
  const paths = series.map((item) => {
    const points = data.map((row, index) => `${xScale(index)},${yScale(toNumber(row[item.key]))}`).join(" ");
    return `<polyline class="${item.className || "line-blue"}" points="${points}"></polyline>`;
  }).join("");
  const labels = series.map((item, index) => `<text x="${pad.left + index * 118}" y="${height - 8}">${escapeHtml(item.label)}</text>`).join("");
  svg.setAttribute("viewBox", `0 0 ${width} ${height}`);
  svg.innerHTML = `${grid}<line class="axis" x1="${pad.left}" y1="${height - pad.bottom}" x2="${width - pad.right}" y2="${height - pad.bottom}"></line>${paths}${labels}`;
}

function drawCalibration(svgId, rows) {
  const svg = $(svgId);
  const width = 720;
  const height = 300;
  const pad = { left: 50, right: 24, top: 20, bottom: 42 };
  const x = (value) => pad.left + clip(value, 0, 1) * (width - pad.left - pad.right);
  const y = (value) => pad.top + (1 - clip(value, 0, 1)) * (height - pad.top - pad.bottom);
  const points = (rows || []).map((row) => `${x(row.mean_confidence)},${y(row.accuracy)}`).join(" ");
  const circles = (rows || []).map((row) => `<circle cx="${x(row.mean_confidence)}" cy="${y(row.accuracy)}" r="${clip(Math.sqrt(toNumber(row.count)) / 2, 4, 15)}" fill="rgba(46,230,138,0.72)"><title>${escapeHtml(row.bin || row.bucket)}: ${fmtPct(row.accuracy)}</title></circle>`).join("");
  svg.setAttribute("viewBox", `0 0 ${width} ${height}`);
  svg.innerHTML = `
    <line class="grid" x1="${pad.left}" y1="${y(0.25)}" x2="${width - pad.right}" y2="${y(0.25)}"></line>
    <line class="grid" x1="${pad.left}" y1="${y(0.5)}" x2="${width - pad.right}" y2="${y(0.5)}"></line>
    <line class="grid" x1="${pad.left}" y1="${y(0.75)}" x2="${width - pad.right}" y2="${y(0.75)}"></line>
    <line class="axis" x1="${pad.left}" y1="${height - pad.bottom}" x2="${width - pad.right}" y2="${height - pad.bottom}"></line>
    <line class="axis" x1="${pad.left}" y1="${pad.top}" x2="${pad.left}" y2="${height - pad.bottom}"></line>
    <line x1="${x(0)}" y1="${y(0)}" x2="${x(1)}" y2="${y(1)}" stroke="rgba(145,165,183,0.65)" stroke-dasharray="6 6"></line>
    <polyline class="line-green" points="${points}"></polyline>
    ${circles}
    <text x="${pad.left}" y="${height - 8}">Mean predicted probability</text>
    <text x="${pad.left}" y="14">Observed accuracy</text>
  `;
}

function drawBankroll(svgId, rows) {
  const data = (rows || []).filter((row, index) => index % 12 === 0 || index === rows.length - 1);
  drawLineChart(svgId, data, [
    { key: "bankroll_after", label: "Bankroll", className: "line-blue" },
  ], {});
}

function drawRadar(svgId, homeName, awayName) {
  const home = teamProfile(homeName);
  const away = teamProfile(awayName);
  const svg = $(svgId);
  if (!home || !away || !svg) {
    if (svg) svg.innerHTML = "";
    return;
  }
  const metrics = [
    ["Elo", "elo", 1300, 1900, false],
    ["Attack", "attack_form", 0.4, 4.5, false],
    ["Defence", "defense_form", 0.4, 3.8, true],
    ["xG diff", "xg_diff_per_match", -1.5, 2.2, false],
    ["Win rate", "win_rate", 0, 0.85, false],
    ["Recent", "recent_points", 0, 3, false],
  ];
  const width = 420;
  const height = 340;
  const cx = width / 2;
  const cy = height / 2;
  const radius = 118;
  const score = (team, metric) => {
    const value = clip((toNumber(team[metric[1]]) - metric[2]) / ((metric[3] - metric[2]) || 1), 0, 1);
    return metric[4] ? 1 - value : value;
  };
  const point = (index, value) => {
    const angle = (-Math.PI / 2) + (index / metrics.length) * Math.PI * 2;
    return [cx + Math.cos(angle) * radius * value, cy + Math.sin(angle) * radius * value];
  };
  const polygon = (team) => metrics.map((metric, index) => point(index, score(team, metric)).join(",")).join(" ");
  const axes = metrics.map((metric, index) => {
    const [x2, y2] = point(index, 1);
    const [lx, ly] = point(index, 1.15);
    return `<line class="grid" x1="${cx}" y1="${cy}" x2="${x2}" y2="${y2}"></line><text x="${lx - 20}" y="${ly + 4}">${escapeHtml(metric[0])}</text>`;
  }).join("");
  const rings = [0.33, 0.66, 1].map((ratio) => `<circle class="grid" cx="${cx}" cy="${cy}" r="${radius * ratio}" fill="none"></circle>`).join("");
  svg.setAttribute("viewBox", `0 0 ${width} ${height}`);
  svg.innerHTML = `
    ${rings}${axes}
    <polygon points="${polygon(home)}" fill="rgba(53,167,255,0.22)" stroke="var(--blue)" stroke-width="3"></polygon>
    <polygon points="${polygon(away)}" fill="rgba(46,230,138,0.16)" stroke="var(--green)" stroke-width="3"></polygon>
    <text x="14" y="${height - 28}">${escapeHtml(homeName)}</text>
    <text x="14" y="${height - 10}">${escapeHtml(awayName)}</text>
  `;
}

function drawTeamTrend(team) {
  const rows = (team.trend || []).map((row) => ({
    date: row.date,
    goals_for: toNumber(row.goals_for),
    goals_against: toNumber(row.goals_against),
    goal_diff: toNumber(row.goal_diff),
  }));
  drawLineChart("teamTrendChart", rows, [
    { key: "goals_for", label: "Goals for", className: "line-blue" },
    { key: "goals_against", label: "Goals against", className: "line-amber" },
    { key: "goal_diff", label: "Goal diff", className: "line-green" },
  ], {});
}

function drawLeagueQuadrant(league) {
  const svg = $("leagueQuadrant");
  const width = 460;
  const height = 340;
  const pad = 42;
  const rows = league.standings || [];
  const maxPower = Math.max(...rows.map((row) => toNumber(row.power_score)), 1);
  const goalDiffs = rows.map((row) => toNumber(row.goal_diff));
  const minGd = Math.min(...goalDiffs, -10);
  const maxGd = Math.max(...goalDiffs, 10);
  const x = (power) => pad + (toNumber(power) / maxPower) * (width - pad * 2);
  const y = (gd) => pad + (1 - ((toNumber(gd) - minGd) / ((maxGd - minGd) || 1))) * (height - pad * 2);
  const circles = rows.slice(0, 20).map((row) => `
    <g>
      <circle cx="${x(row.power_score)}" cy="${y(row.goal_diff)}" r="7" fill="rgba(53,167,255,0.76)"></circle>
      <title>${escapeHtml(row.team)} power ${fmtNum(row.power_score, 1)} GD ${fmtInt(row.goal_diff)}</title>
    </g>
  `).join("");
  svg.setAttribute("viewBox", `0 0 ${width} ${height}`);
  svg.innerHTML = `
    <line class="axis" x1="${pad}" y1="${height / 2}" x2="${width - pad}" y2="${height / 2}"></line>
    <line class="axis" x1="${width / 2}" y1="${pad}" x2="${width / 2}" y2="${height - pad}"></line>
    ${circles}
    <text x="${pad}" y="${height - 10}">Power score</text>
    <text x="${pad}" y="22">Goal difference</text>
  `;
}

function renderCharts() {
  if (!state.report) return;
  renderDashboard();
  renderMatch();
  renderTeams();
  renderLeagues();
  renderBacktest();
}

function exportPredictionsCsv() {
  const rows = state.fixtures;
  const csv = toCsv(rows, [
    { label: "date", value: (row) => row.date },
    { label: "competition", value: (row) => row.competition },
    { label: "home_team", value: (row) => row.home_team },
    { label: "away_team", value: (row) => row.away_team },
    { label: "p_home", value: (row) => row.probabilities?.home },
    { label: "p_draw", value: (row) => row.probabilities?.draw },
    { label: "p_away", value: (row) => row.probabilities?.away },
    { label: "predicted_score", value: (row) => row.predicted_score },
    { label: "home_xg", value: (row) => row.expected_goals?.home },
    { label: "away_xg", value: (row) => row.expected_goals?.away },
    { label: "confidence", value: (row) => row.confidence?.score },
    { label: "best_edge", value: (row) => bestEdge(row, true) },
  ]);
  downloadText("football-predictions.csv", csv, "text/csv");
}

function exportBacktestCsv() {
  const rows = state.report.backtest_dashboard?.archive || [];
  const csv = toCsv(rows, [
    { label: "date", value: (row) => row.date },
    { label: "competition", value: (row) => row.competition },
    { label: "home_team", value: (row) => row.home_team },
    { label: "away_team", value: (row) => row.away_team },
    { label: "home_goals", value: (row) => row.home_goals },
    { label: "away_goals", value: (row) => row.away_goals },
    { label: "p_home", value: (row) => row.p_home },
    { label: "p_draw", value: (row) => row.p_draw },
    { label: "p_away", value: (row) => row.p_away },
    { label: "top_score", value: (row) => row.top_score },
  ]);
  downloadText("football-backtest-archive.csv", csv, "text/csv");
}

function openSearch() {
  $("commandPalette").classList.add("open");
  $("commandPalette").setAttribute("aria-hidden", "false");
  $("commandInput").value = "";
  renderSearchResults();
  setTimeout(() => $("commandInput").focus(), 20);
}

function closeSearch() {
  $("commandPalette").classList.remove("open");
  $("commandPalette").setAttribute("aria-hidden", "true");
}

function renderSearchResults() {
  const query = $("commandInput").value.trim().toLowerCase();
  const fixtures = state.fixtures
    .filter((fixture) => `${fixture.home_team} ${fixture.away_team} ${fixture.competition}`.toLowerCase().includes(query))
    .slice(0, 8)
    .map((fixture) => ({ type: "Match", title: `${fixture.home_team} vs ${fixture.away_team}`, detail: `${fixture.date} - ${fixture.competition}`, action: () => setSelectedMatch(fixture, true) }));
  const teams = (state.report.team_analytics || [])
    .filter((team) => team.team.toLowerCase().includes(query))
    .slice(0, 6)
    .map((team) => ({ type: "Team", title: team.team, detail: `Elo ${fmtInt(team.elo)} - ${team.form}`, action: () => {
      state.selectedTeam = team.team;
      $("teamSelect").value = team.team;
      renderTeams();
      window.location.hash = "teams";
      $("teams").scrollIntoView({ behavior: "smooth", block: "start" });
    } }));
  const leagues = (state.report.league_analytics || [])
    .filter((league) => league.competition.toLowerCase().includes(query))
    .slice(0, 6)
    .map((league) => ({ type: "League", title: league.competition, detail: `${fmtInt(league.matches)} matches - accuracy ${fmtPct(league.accuracy)}`, action: () => {
      state.selectedLeague = league.competition;
      $("leagueSelect").value = league.competition;
      renderLeagues();
      window.location.hash = "leagues";
      $("leagues").scrollIntoView({ behavior: "smooth", block: "start" });
    } }));
  const results = [...fixtures, ...teams, ...leagues].slice(0, 18);
  $("commandResults").innerHTML = results.length ? results.map((item, index) => `
    <button class="command-result" data-search-index="${index}" type="button">
      <span><strong>${escapeHtml(item.title)}</strong><small>${escapeHtml(item.detail)}</small></span>
      <span class="pill">${escapeHtml(item.type)}</span>
    </button>
  `).join("") : emptyState("No results.");
  $("commandResults").querySelectorAll("[data-search-index]").forEach((button) => {
    button.addEventListener("click", () => {
      const item = results[Number(button.dataset.searchIndex)];
      closeSearch();
      item.action();
    });
  });
}

function updateActiveNav() {
  const active = window.location.hash.replace("#", "") || "dashboard";
  document.querySelectorAll("[data-nav]").forEach((link) => {
    link.classList.toggle("active", link.dataset.nav === active);
  });
}

function scrollToHashTarget() {
  const active = window.location.hash.replace("#", "") || "dashboard";
  const target = $(active);
  if (target) target.scrollIntoView({ behavior: "auto", block: "start" });
}

async function init() {
  setupStaticEvents();
  if ("scrollRestoration" in history) history.scrollRestoration = "manual";
  const savedTheme = localStorage.getItem(STORAGE_KEYS.theme);
  if (savedTheme) document.documentElement.dataset.theme = savedTheme;
  loadWatchlist();

  try {
    const response = await fetch("data/report.json", { cache: "no-store" });
    if (!response.ok) throw new Error(`Report fetch failed: ${response.status}`);
    state.report = await response.json();
    state.fixtures = buildFixtures(state.report);
    state.selectedMatch = state.report.match_detail || state.fixtures[0] || null;
    populateControls();
    setupDataEvents();
    renderStatus();
    renderDashboard();
    renderFixtures();
    renderMatch();
    renderTeams();
    renderLeagues();
    renderBacktest();
    renderLab();
    updateActiveNav();
    requestAnimationFrame(scrollToHashTarget);
  } catch (error) {
    console.error(error);
    document.querySelector(".main-shell").innerHTML = `<div class="panel">${emptyState("Report could not be loaded. Regenerate web/data/report.json and refresh.")}</div>`;
  }
}

init();
