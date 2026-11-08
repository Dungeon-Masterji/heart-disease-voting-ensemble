interface Point { x: number; y: number }

interface Props {
  title: string;
  points: Point[];
  threshold: number;
  xLabel: string;
  yLabel: string;
  summary: string;
}

export default function CurvesChart({ title, points, threshold, xLabel, yLabel, summary }: Props) {
  const width = 520;
  const height = 260;
  const pad = 42;
  const coords = points.map((point) => ({
    x: pad + point.x * (width - pad * 2),
    y: height - pad - point.y * (height - pad * 2),
  }));
  const polyline = coords.map(({ x, y }) => `${x},${y}`).join(" ");
  const markerX = pad + threshold * (width - pad * 2);
  const titleId = `${title.replace(/[^a-z0-Z]/g, "-")}-title`;

  return (
    <figure className="overflow-x-auto rounded border border-slate-300 bg-white p-4">
      <h3 className="mb-2 font-semibold">{title}</h3>
      <svg viewBox={`0 0 ${width} ${height}`} role="img" aria-labelledby={titleId}>
        <title id={titleId}>{title}</title>
        <desc>{summary} The operating threshold is {threshold.toFixed(3)}.</desc>
        <line x1={pad} y1={height - pad} x2={width - pad} y2={height - pad} stroke="#334155" />
        <line x1={pad} y1={pad} x2={pad} y2={height - pad} stroke="#334155" />
        <polyline points={polyline} fill="none" stroke="#1d4ed8" strokeWidth="3" />
        <line x1={markerX} y1={pad} x2={markerX} y2={height - pad} stroke="#b91c1c" strokeDasharray="5 4" />
        <text x={width / 2} y={height - 8} textAnchor="middle" fontSize="12">{xLabel}</text>
        <text x="12" y={height / 2} textAnchor="middle" fontSize="12" transform={`rotate(-90 12 ${height / 2})`}>{yLabel}</text>
      </svg>
      <figcaption className="mt-2 text-sm text-slate-700">
        {summary} Operating threshold: {threshold.toFixed(3)}.
      </figcaption>
    </figure>
  );
}
