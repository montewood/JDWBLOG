(() => {
  const dataUrlTemplate =
    "https://raw.githubusercontent.com/montewood/gh-action/refs/heads/main/output/GA-DATE_KEY.json";
  const numberOfDays = 90;
  const concurrency = 6;

  const formatDateKey = (date) => {
    const year = date.getFullYear();
    const month = String(date.getMonth() + 1).padStart(2, "0");
    const day = String(date.getDate()).padStart(2, "0");
    return `${year}${month}${day}`;
  };

  const dateKeys = () => {
    const endDate = new Date();
    endDate.setHours(0, 0, 0, 0);
    endDate.setDate(endDate.getDate() - 1);

    return Array.from({ length: numberOfDays }, (_, index) => {
      const date = new Date(endDate);
      date.setDate(endDate.getDate() - (numberOfDays - index - 1));
      return formatDateKey(date);
    });
  };

  const unwrapRows = (value) => {
    if (
      value &&
      !Array.isArray(value) &&
      Number.isFinite(Number(value.activeUsers))
    ) {
      return [{ activeUsers: Number(value.activeUsers) }];
    }

    let currentValue = value;

    for (let depth = 0; depth < 4; depth += 1) {
      if (typeof currentValue === "string") {
        currentValue = JSON.parse(currentValue);
      } else if (
        Array.isArray(currentValue) &&
        currentValue.length === 1 &&
        (Array.isArray(currentValue[0]) || typeof currentValue[0] === "string")
      ) {
        currentValue = currentValue[0];
      } else {
        break;
      }
    }

    return Array.isArray(currentValue) ? currentValue : [];
  };

  const fetchDay = async (dateKey) => {
    try {
      const response = await fetch(dataUrlTemplate.replace("DATE_KEY", dateKey), {
        headers: { Accept: "application/json" },
      });

      if (!response.ok) {
        return { date: dateKey, value: 0, status: response.status === 404 ? "missing" : "error" };
      }

      const rows = unwrapRows(await response.json());
      const value = rows.reduce((sum, row) => {
        const activeUsers = Number(row?.activeUsers);
        return sum + (Number.isFinite(activeUsers) ? activeUsers : 0);
      }, 0);

      return { date: dateKey, value, status: rows.length ? "ok" : "empty" };
    } catch {
      return { date: dateKey, value: 0, status: "error" };
    }
  };

  const loadAnalytics = async () => {
    const keys = dateKeys();
    const results = new Array(keys.length);
    let nextIndex = 0;
    let completed = 0;

    const worker = async () => {
      while (nextIndex < keys.length) {
        const currentIndex = nextIndex;
        nextIndex += 1;
        results[currentIndex] = await fetchDay(keys[currentIndex]);
        completed += 1;
        window.Shiny.setInputValue(
          "analytics_progress",
          { completed, total: keys.length },
          { priority: "event" },
        );
      }
    };

    await Promise.all(
      Array.from({ length: Math.min(concurrency, keys.length) }, worker),
    );
    window.Shiny.setInputValue(
      "analytics_payload",
      {
        dates: results.map(({ date }) => date),
        values: results.map(({ value }) => value),
        statuses: results.map(({ status }) => status),
      },
      { priority: "event" },
    );
  };

  if (window.jQuery) {
    window.jQuery(document).one("shiny:connected", loadAnalytics);
  } else {
    document.addEventListener("shiny:connected", loadAnalytics, { once: true });
  }
})();
