import React from "react";
import ReactDOM from "react-dom/client";
import "./styles/index.css";
import { AppShell } from "./components/layout/AppShell";

ReactDOM.createRoot(document.getElementById("root") as HTMLElement).render(
  <React.StrictMode>
    <AppShell />
  </React.StrictMode>
);

