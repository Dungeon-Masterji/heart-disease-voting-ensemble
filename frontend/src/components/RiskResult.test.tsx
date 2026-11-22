import { render, screen } from "@testing-library/react";
import RiskResult from "./RiskResult";
import type { ScoreResponse } from "../types";

const base: ScoreResponse = {
  risk_probability: 0.72, risk_flag: "High Risk", model_version: "m2.0.0", voting_strategy: "soft",
  threshold: 0.4, model_used: "full", missing_tests: [], notice: null, timestamp: "2026-01-01T00:00:00Z", disclaimer: "Clinical decision-support signal, not a medical diagnosis.",
};

describe("RiskResult", () => {
  it("shows the flag as text, the probability, and the model version", () => {
    render(<RiskResult result={base} />);
    expect(screen.getByText("High Risk")).toBeInTheDocument();
    expect(screen.getByText("m2.0.0")).toBeInTheDocument();
    expect(screen.getByRole("progressbar")).toHaveAttribute("aria-valuenow", "72");
  });
  it("shows Low Risk and always shows a disclaimer", () => {
    render(<RiskResult result={{ ...base, risk_flag: "Low Risk", risk_probability: 0.1 }} />);
    expect(screen.getByText("Low Risk")).toBeInTheDocument();
    expect(screen.getAllByText(/not a medical diagnosis/i).length).toBeGreaterThan(0);
  });
  it("shows the notice when a scan is missing, and only then", () => {
    const { rerender } = render(<RiskResult result={base} />);
    expect(screen.queryByRole("note")).toBeNull();
    rerender(<RiskResult result={{ ...base, model_used: "no_scan", missing_tests: ["ca"], notice: "Less certain." }} />);
    expect(screen.getByRole("note")).toHaveTextContent("Less certain. Not available: ca.");
  });
});
