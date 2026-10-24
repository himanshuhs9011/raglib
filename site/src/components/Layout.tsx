import type { ReactNode } from "react";
import { Logo } from "./Logo";
import { LINKS, VERSION } from "../data";
import type { Route } from "../App";

const NAV: { route: Route; label: string }[] = [
  { route: "home", label: "Home" },
  { route: "reference", label: "API" },
  { route: "docs", label: "Docs" },
  { route: "playground", label: "Playground" },
];

export function Layout({ route, children }: { route: Route; children: ReactNode }) {
  return (
    <>
      <header className="site-header">
        <div className="wrap">
          <a className="brand" href="#/">
            <Logo size={26} />
            raglib <span className="ver">v{VERSION}</span>
          </a>
          <nav className="nav" aria-label="Main">
            {NAV.map((item) => (
              <a
                key={item.route}
                href={item.route === "home" ? "#/" : `#/${item.route}`}
                aria-current={route === item.route ? "page" : undefined}
              >
                {item.label}
              </a>
            ))}
            <a href={LINKS.github} target="_blank" rel="noreferrer">GitHub</a>
          </nav>
        </div>
      </header>
      <main>{children}</main>
      <footer className="site-footer">
        <div className="wrap">
          <span>raglib-py {VERSION} · MIT licence · Python 3.9+</span>
          <nav aria-label="Footer">
            <a href={LINKS.pypi} target="_blank" rel="noreferrer">PyPI</a>
            <a href={LINKS.github} target="_blank" rel="noreferrer">GitHub</a>
            <a href={LINKS.readme} target="_blank" rel="noreferrer">README</a>
            <a href={LINKS.changelog} target="_blank" rel="noreferrer">Changelog</a>
            <a href={LINKS.issues} target="_blank" rel="noreferrer">Issues</a>
          </nav>
        </div>
      </footer>
    </>
  );
}
