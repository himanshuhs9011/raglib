/** Square, pixel-grid "R" mark (retro, black and white). */
export function Logo({ size = 28, className }: { size?: number; className?: string }) {
  return (
    <svg
      className={className}
      width={size}
      height={size}
      viewBox="0 0 32 32"
      shapeRendering="crispEdges"
      role="img"
      aria-label="raglib logo"
    >
      <rect width="32" height="32" fill="#000" />
      <rect x="2" y="2" width="28" height="28" fill="none" stroke="#fff" strokeWidth="2" />
      <g fill="#fff">
        <rect x="9" y="8" width="10" height="3" />
        <rect x="9" y="11" width="3" height="13" />
        <rect x="19" y="11" width="3" height="5" />
        <rect x="12" y="16" width="7" height="3" />
        <rect x="16" y="19" width="3" height="2" />
        <rect x="19" y="21" width="3" height="3" />
      </g>
    </svg>
  );
}
