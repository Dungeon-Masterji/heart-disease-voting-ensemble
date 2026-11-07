import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import CurvesChart from "./CurvesChart";

describe("CurvesChart", () => {
  it("renders an accessible inline SVG with the threshold summary", () => {
    render(
      <CurvesChart
        title="ROC curve"
        points={[{ x: 0, y: 0 }, { x: 0.5, y: 0.8 }, { x: 1, y: 1 }]}
        threshold={0.5}
        xLabel="False positive rate"
        yLabel="True positive rate"
        summary="Out-of-fold ROC curve."
      />,
    );
    expect(screen.getByRole("img", { name: "ROC curve" })).toBeInTheDocument();
    expect(screen.getByText(/operating threshold is 0.500/i)).toBeInTheDocument();
  });
});
