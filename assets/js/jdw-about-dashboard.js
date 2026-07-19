import * as echarts from "echarts/core";
import { BarChart } from "echarts/charts";
import { GridComponent, TooltipComponent } from "echarts/components";
import { CanvasRenderer } from "echarts/renderers";

echarts.use([BarChart, GridComponent, TooltipComponent, CanvasRenderer]);

const REQUEST_CONCURRENCY = 6;
const SESSION_ID_KEY = "jdw-dashboard-session-id";
const ACCESS_CACHE_KEY = "jdw-dashboard-access";
const dayCache = new Map();

const formatDateKey = (date) => {
  const year = date.getFullYear();
  const month = String(date.getMonth() + 1).padStart(2, "0");
  const day = String(date.getDate()).padStart(2, "0");
  return `${year}${month}${day}`;
};

const formatDateLabel = (dateKey) =>
  `${dateKey.slice(0, 4)}-${dateKey.slice(4, 6)}-${dateKey.slice(6, 8)}`;

const buildDateKeys = (numberOfDays) => {
  const endDate = new Date();
  endDate.setHours(0, 0, 0, 0);
  endDate.setDate(endDate.getDate() - 1);

  return Array.from({ length: numberOfDays }, (_, index) => {
    const date = new Date(endDate);
    date.setDate(endDate.getDate() - (numberOfDays - index - 1));
    return formatDateKey(date);
  });
};

const parseDailyValue = (value) => {
  if (
    value &&
    !Array.isArray(value) &&
    Number.isFinite(Number(value.activeUsers))
  ) {
    return Number(value.activeUsers);
  }

  let currentValue = value;

  for (let depth = 0; depth < 4; depth += 1) {
    if (typeof currentValue === "string") {
      currentValue = JSON.parse(currentValue);
      continue;
    }

    if (Array.isArray(currentValue) && currentValue.length === 1 && Array.isArray(currentValue[0])) {
      currentValue = currentValue[0];
      continue;
    }

    if (Array.isArray(currentValue) && currentValue.length === 1 && typeof currentValue[0] === "string") {
      currentValue = currentValue[0];
      continue;
    }

    break;
  }

  if (!Array.isArray(currentValue)) {
    throw new TypeError("Daily analytics payload is not supported.");
  }
  if (currentValue.length === 0) return null;

  return currentValue.reduce((sum, row) => {
    const activeUsers = Number(row?.activeUsers);
    return sum + (Number.isFinite(activeUsers) ? activeUsers : 0);
  }, 0);
};

const loadDay = (dateKey, urlTemplate) => {
  const cacheKey = `${urlTemplate}:${dateKey}`;

  if (!dayCache.has(cacheKey)) {
    const request = fetch(urlTemplate.replace("DATE_KEY", dateKey), {
      headers: { Accept: "application/json" },
    }).then(async (response) => {
      if (response.status === 404) {
        return { dateKey, status: "missing", value: 0 };
      }

      if (!response.ok) {
        throw new Error(`Analytics request failed with ${response.status}.`);
      }

      const payload = await response.json();
      const value = parseDailyValue(payload);

      return value === null
        ? { dateKey, status: "empty", value: 0 }
        : {
            dateKey,
            status: "ok",
            value,
            schema: payload?.schemaVersion === 2 ? "daily-users" : "legacy",
          };
    }).catch((error) => ({
      dateKey,
      status: "error",
      value: 0,
      error,
    }));

    dayCache.set(cacheKey, request);
  }

  return dayCache.get(cacheKey);
};

const loadDays = async (dateKeys, urlTemplate, onProgress) => {
  const results = new Array(dateKeys.length);
  let nextIndex = 0;
  let completed = 0;

  const worker = async () => {
    while (nextIndex < dateKeys.length) {
      const currentIndex = nextIndex;
      nextIndex += 1;
      results[currentIndex] = await loadDay(dateKeys[currentIndex], urlTemplate);
      completed += 1;
      onProgress(completed, dateKeys.length);
    }
  };

  const workerCount = Math.min(REQUEST_CONCURRENCY, dateKeys.length);
  await Promise.all(Array.from({ length: workerCount }, worker));
  return results;
};

const getThemeColors = (element) => {
  const styles = getComputedStyle(element);
  return {
    text: styles.getPropertyValue("--jdw-text").trim() || "#1f2937",
    primary: styles.getPropertyValue("--jdw-primary").trim() || "#2563eb",
    grid: `color-mix(in srgb, ${styles.getPropertyValue("--jdw-text").trim() || "#1f2937"} 14%, transparent)`,
  };
};

const getSessionId = () => {
  let sessionId = sessionStorage.getItem(SESSION_ID_KEY);
  if (!sessionId) {
    sessionId = crypto.randomUUID();
    sessionStorage.setItem(SESSION_ID_KEY, sessionId);
  }
  return sessionId;
};

const getCachedAccess = () => {
  try {
    const cached = JSON.parse(sessionStorage.getItem(ACCESS_CACHE_KEY) || "null");
    return cached?.appUrl && cached?.expiresAt > Date.now() ? cached : null;
  } catch {
    return null;
  }
};

const cacheAccess = ({ appUrl, expiresAt }) => {
  sessionStorage.setItem(
    ACCESS_CACHE_KEY,
    JSON.stringify({ appUrl, expiresAt }),
  );
};

const isLocalPreview = () =>
  ["localhost", "127.0.0.1"].includes(window.location.hostname);

const updateQuotaUi = (root, quota) => {
  const openButton = root.querySelector("[data-dashboard-open]");
  const quotaLabel = root.querySelector("[data-dashboard-quota]");
  const quotaStatus = root.querySelector("[data-dashboard-quota-status]");
  if (!openButton || !quotaLabel || !quotaStatus) return;

  if (quota.mode === "development") {
    openButton.disabled = false;
    quotaLabel.textContent = "개발 모드";
    quotaStatus.textContent = "로컬 미리보기에서는 실행 쿼터를 차감하지 않습니다.";
    return;
  }

  if (quota.error) {
    openButton.disabled = true;
    quotaLabel.textContent = "실행 불가";
    quotaStatus.textContent = quota.error;
    return;
  }

  openButton.disabled = !quota.available;
  quotaLabel.textContent = quota.available
    ? `${quota.remaining}/${quota.limit}`
    : `마감 · 0/${quota.limit}`;
  quotaStatus.textContent = quota.available
    ? `오늘 ${quota.remaining}회 실행할 수 있습니다. 같은 탭 세션의 재실행은 차감되지 않습니다.`
    : "오늘의 대시보드 실행 횟수가 모두 사용되었습니다.";
};

const fetchQuota = async (root) => {
  try {
    const endpoint = new URL(root.dataset.launchEndpoint, window.location.origin);
    endpoint.searchParams.set("session_id", getSessionId());
    const response = await fetch(endpoint, {
      headers: { Accept: "application/json" },
    });
    if (!response.ok) throw new Error("실행 횟수를 확인하지 못했습니다.");
    const quota = await response.json();
    root.dataset.quotaConfigured = "true";
    updateQuotaUi(root, quota);
    return quota;
  } catch {
    if (isLocalPreview()) {
      const developmentQuota = { mode: "development", available: true };
      updateQuotaUi(root, developmentQuota);
      return developmentQuota;
    }
    const failedQuota = {
      error: "대시보드 실행 서비스를 확인해 주세요.",
      available: false,
    };
    updateQuotaUi(root, failedQuota);
    return failedQuota;
  }
};

const requestLaunch = async (root) => {
  const cached = getCachedAccess();
  if (cached) return cached.appUrl;
  if (isLocalPreview() && root.dataset.quotaConfigured !== "true") {
    return root.dataset.appUrl;
  }

  const response = await fetch(root.dataset.launchEndpoint, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      Accept: "application/json",
    },
    body: JSON.stringify({ sessionId: getSessionId() }),
  });
  const result = await response.json();
  updateQuotaUi(root, {
    ...result,
    available: result.remaining > 0 || result.allowed,
  });

  if (!response.ok || !result.allowed || !result.appUrl) {
    throw new Error(
      response.status === 429
        ? "오늘의 대시보드 실행 횟수가 모두 사용되었습니다."
        : "대시보드 실행 권한을 발급하지 못했습니다.",
    );
  }

  cacheAccess(result);
  return result.appUrl;
};

const initializeModal = (root) => {
  const dialog = root.querySelector("[data-dashboard-dialog]");
  const openButton = root.querySelector("[data-dashboard-open]");
  const launchLabel = root.querySelector("[data-dashboard-launch-label]");
  const quotaStatus = root.querySelector("[data-dashboard-quota-status]");
  const closeButton = root.querySelector("[data-dashboard-close]");
  const frameHost = root.querySelector("[data-dashboard-frame-host]");
  const loadingStatus = root.querySelector("[data-dashboard-frame-status]");

  if (
    !dialog ||
    !openButton ||
    !launchLabel ||
    !quotaStatus ||
    !closeButton ||
    !frameHost ||
    !loadingStatus
  ) return;

  const removeFrame = () => {
    frameHost.querySelector("iframe")?.remove();
    loadingStatus.hidden = false;
    openButton.focus();
  };

  openButton.addEventListener("click", async () => {
    const defaultLabel = root.dataset.appButtonLabel || "대시보드 실행";
    openButton.disabled = true;
    launchLabel.textContent = "실행 준비 중";

    try {
      const appUrl = await requestLaunch(root);
      const frame = document.createElement("iframe");
      frame.title = root.dataset.appTitle || "R 기반 방문자 대시보드";
      frame.src = appUrl;
      frame.loading = "eager";
      frame.setAttribute(
        "sandbox",
        "allow-downloads allow-forms allow-modals allow-popups allow-same-origin allow-scripts",
      );
      frame.addEventListener("load", () => {
        loadingStatus.hidden = true;
      }, { once: true });
      frameHost.append(frame);
      dialog.showModal();
    } catch (error) {
      quotaStatus.textContent = error instanceof Error
        ? error.message
        : "대시보드를 실행하지 못했습니다.";
    } finally {
      launchLabel.textContent = defaultLabel;
      await fetchQuota(root);
    }
  });

  closeButton.addEventListener("click", () => dialog.close());
  dialog.addEventListener("close", removeFrame);
  dialog.addEventListener("click", (event) => {
    if (event.target === dialog) dialog.close();
  });

  fetchQuota(root);
};

const initializeDashboard = (root) => {
  if (root.dataset.initialized === "true") return;
  root.dataset.initialized = "true";

  const chartElement = root.querySelector("[data-dashboard-chart]");
  const rangeSelect = root.querySelector("[data-dashboard-range]");
  const statusElement = root.querySelector("[data-dashboard-status]");
  const latestElement = root.querySelector("[data-dashboard-latest]");
  const averageElement = root.querySelector("[data-dashboard-average]");
  const coverageElement = root.querySelector("[data-dashboard-coverage]");

  if (
    !chartElement ||
    !rangeSelect ||
    !statusElement ||
    !latestElement ||
    !averageElement ||
    !coverageElement
  ) {
    return;
  }

  const chart = echarts.init(chartElement, null, { renderer: "canvas" });
  let renderSequence = 0;
  let latestResults = [];

  const renderChart = (results) => {
    const colors = getThemeColors(root);
    const successfulResults = results.filter(({ status }) => status === "ok");

    chart.setOption({
      animationDuration: 350,
      color: [colors.primary],
      grid: { left: 54, right: 16, top: 28, bottom: 54 },
      tooltip: {
        trigger: "axis",
        valueFormatter: (value) => `${Number(value).toLocaleString("ko-KR")}회`,
      },
      xAxis: {
        type: "category",
        data: results.map(({ dateKey }) => formatDateLabel(dateKey).slice(5)),
        axisLabel: { color: colors.text, hideOverlap: true },
        axisLine: { lineStyle: { color: colors.grid } },
      },
      yAxis: {
        type: "value",
        name: "Active Users",
        minInterval: 1,
        nameTextStyle: { color: colors.text },
        axisLabel: { color: colors.text },
        splitLine: { lineStyle: { color: colors.grid } },
      },
      series: [{
        name: "일별 Active Users",
        type: "bar",
        data: results.map(({ status, value }) => status === "ok" ? value : 0),
        barMaxWidth: 18,
      }],
    }, true);

    const total = successfulResults.reduce((sum, { value }) => sum + value, 0);
    const average = successfulResults.length ? total / successfulResults.length : 0;
    const unavailable = results.length - successfulResults.length;
    const latest = successfulResults.at(-1);

    latestElement.textContent = latest
      ? latest.value.toLocaleString("ko-KR")
      : "—";
    averageElement.textContent = average.toLocaleString("ko-KR", { maximumFractionDigits: 1 });
    coverageElement.textContent = `${successfulResults.length}/${results.length}`;
    chartElement.setAttribute(
      "aria-label",
      `최근 ${results.length}일 고유 Active Users 막대그래프. 수집된 날짜 ${successfulResults.length}일, 누락 또는 실패 ${unavailable}일.`,
    );
  };

  const refresh = async () => {
    const currentSequence = ++renderSequence;
    const numberOfDays = Number(rangeSelect.value || root.dataset.defaultDays || 90);
    const dateKeys = buildDateKeys(numberOfDays);
    const urlTemplate = root.dataset.dataUrlTemplate;

    rangeSelect.disabled = true;
    statusElement.textContent = `일별 데이터 0/${numberOfDays} 불러오는 중`;

    const results = await loadDays(dateKeys, urlTemplate, (completed, total) => {
      if (currentSequence === renderSequence) {
        statusElement.textContent = `일별 데이터 ${completed}/${total} 불러오는 중`;
      }
    });

    if (currentSequence !== renderSequence) return;

    latestResults = results;
    const successCount = results.filter(({ status }) => status === "ok").length;
    const missingCount = results.filter(({ status }) => status === "missing" || status === "empty").length;
    const errorCount = results.filter(({ status }) => status === "error").length;
    const legacyCount = results.filter(({ schema }) => schema === "legacy").length;

    renderChart(results);
    statusElement.textContent = successCount
      ? `${successCount}일 수집 완료${missingCount ? ` · ${missingCount}일 데이터 없음` : ""}${errorCount ? ` · ${errorCount}일 요청 실패` : ""}${legacyCount ? ` · 구형 집계 ${legacyCount}일` : ""}`
      : "표시할 데이터를 불러오지 못했습니다.";
    rangeSelect.disabled = false;
  };

  rangeSelect.addEventListener("change", refresh);
  new ResizeObserver(() => chart.resize()).observe(chartElement);

  const themeObserver = new MutationObserver(() => {
    if (latestResults.length) renderChart(latestResults);
  });
  themeObserver.observe(document.documentElement, {
    attributes: true,
    attributeFilter: ["class", "data-theme"],
  });

  initializeModal(root);
  refresh();
};

document.querySelectorAll("[data-jdw-dashboard]").forEach(initializeDashboard);
