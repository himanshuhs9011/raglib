import { useEffect, useState } from "react";
import { Layout } from "./components/Layout";
import { Home } from "./pages/Home";
import { Docs } from "./pages/Docs";
import { Playground } from "./pages/Playground";
import { Reference } from "./pages/Reference";

export type Route = "home" | "reference" | "docs" | "playground";

const TITLES: Record<Route, string> = {
  home: "raglib: production RAG for Python",
  reference: "API reference · raglib",
  docs: "Docs · raglib",
  playground: "Playground · raglib",
};

/** Hash routes ("#/docs", "#/docs/graphrag") work on any static host with no redirect rules. */
function parseHash(): { route: Route; anchor?: string } {
  const [, page = "", anchor] = window.location.hash.replace(/^#/, "").split("/");
  const route: Route = page === "docs" || page === "playground" || page === "reference" ? page : "home";
  return { route, anchor };
}

export function App() {
  const [{ route, anchor }, setLocation] = useState(parseHash);
  const [playgroundLoaded, setPlaygroundLoaded] = useState(route === "playground");

  useEffect(() => {
    const onChange = () => setLocation(parseHash());
    window.addEventListener("hashchange", onChange);
    return () => window.removeEventListener("hashchange", onChange);
  }, []);

  useEffect(() => {
    document.title = TITLES[route];
    if (route === "playground") setPlaygroundLoaded(true);
    const target = anchor ? document.getElementById(anchor) : null;
    if (target) target.scrollIntoView();
    else window.scrollTo(0, 0);
  }, [route, anchor]);

  return (
    <Layout route={route}>
      {route === "home" && <Home />}
      {route === "reference" && <Reference />}
      {route === "docs" && <Docs />}
      {/* Stays mounted once opened, so the in-browser Python runtime is not reloaded. */}
      {playgroundLoaded && (
        <div hidden={route !== "playground"}>
          <Playground />
        </div>
      )}
    </Layout>
  );
}
