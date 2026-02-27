// Fonction utilitaire pour générer des bougies OHLC simulées pour la démo
window.generateIntroSeries = function (count) {
  const candles = [];
  const resistanceLine = [];

  // Paramètres initiaux
  let currentDate = new Date("2024-01-01T10:00:00Z");
  let currentPrice = 40000;
  let volatility = 100;

  // Définir la ligne de résistance (légèrement baissière au début, puis plate)
  let resistanceStart = 40500;

  for (let i = 0; i < count; i++) {
    // Ajout de temps (1 heure par bougie)
    currentDate.setHours(currentDate.getHours() + 1);
    const time = currentDate.getTime() / 1000;

    // Générer les prix OHLC (Random walk modéré)
    const open = currentPrice;
    const change = (Math.random() - 0.5) * volatility;
    let close = open + change;

    // Simuler la compression sous la résistance
    const resistanceValue = resistanceStart - i * 2; // Pente baissière très légère

    resistanceLine.push({ time, value: resistanceValue });

    // Si on est avant la bougie 80, on écrase le prix s'il dépasse la résistance (rejet)
    if (i < 80) {
      if (close > resistanceValue - 50) {
        close = resistanceValue - 100 - Math.random() * 50; // Rejet
      }
    } else if (i === 80) {
      // Le BREAKOUT (bougie 80)
      close = resistanceValue + 300; // Forte cassure
    } else {
      // Après le breakout, ça monte
      close = open + Math.abs(change) + 50;
    }

    const high = Math.max(open, close) + Math.random() * volatility * 0.5;
    const low = Math.min(open, close) - Math.random() * volatility * 0.5;

    candles.push({ time, open, high, low, close });
    currentPrice = close;
  }

  return {
    candles,
    resistanceLine,
    breakoutTime: candles[80].time, // Time de la bougie de cassure
  };
};
