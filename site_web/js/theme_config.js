// Configuration du thème sombre pour TradingView Lightweight Charts
window.chartDarkTheme = {
  layout: {
    background: {
      type: "solid",
      color:
        getComputedStyle(document.documentElement)
          .getPropertyValue("--bg-card")
          .trim() || "#21262d",
    },
    textColor:
      getComputedStyle(document.documentElement)
        .getPropertyValue("--text-secondary")
        .trim() || "#8b949e",
  },
  grid: {
    vertLines: {
      color:
        getComputedStyle(document.documentElement)
          .getPropertyValue("--border-color")
          .trim() || "#30363d",
    },
    horzLines: {
      color:
        getComputedStyle(document.documentElement)
          .getPropertyValue("--border-color")
          .trim() || "#30363d",
    },
  },
  crosshair: {
    mode: LightweightCharts.CrosshairMode.Normal,
  },
  rightPriceScale: {
    borderColor:
      getComputedStyle(document.documentElement)
        .getPropertyValue("--border-color")
        .trim() || "#30363d",
  },
  timeScale: {
    borderColor:
      getComputedStyle(document.documentElement)
        .getPropertyValue("--border-color")
        .trim() || "#30363d",
    timeVisible: true,
    secondsVisible: false,
  },
};
