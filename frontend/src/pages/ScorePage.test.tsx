import { fireEvent, render, screen } from "@testing-library/react";
import { vi } from "vitest";
import ScorePage from "./ScorePage";
import { scorePatient } from "../services/api";

vi.mock("../services/api", () => ({
  scorePatient: vi.fn(),
  toApiFailure: () => ({ status: 503, code: "MODEL_UNAVAILABLE", message: "down" }),
}));

const mockedScore = vi.mocked(scorePatient);

describe("ScorePage", () => {
  beforeEach(() => mockedScore.mockReset());

  it("shows the empty-state prompt initially", () => {
    render(<ScorePage />);
    expect(screen.getByText(/Enter the patient/i)).toBeInTheDocument();
  });

  it("shows inline validation errors and does not call the API for an empty form", () => {
    render(<ScorePage />);
    fireEvent.click(screen.getByRole("button", { name: /get risk flag/i }));
    expect(screen.getAllByText("Required").length).toBeGreaterThan(0);
    expect(mockedScore).not.toHaveBeenCalled();
  });

  it("submits the sample patient and renders the result", async () => {
    mockedScore.mockResolvedValue({
      risk_probability: 0.8, risk_flag: "High Risk", model_version: "m1.0.0", voting_strategy: "soft",
      threshold: 0.4, timestamp: "2026-01-01T00:00:00Z", disclaimer: "Not a diagnosis.",
    });
    render(<ScorePage />);
    fireEvent.click(screen.getByRole("button", { name: /load sample patient/i }));
    fireEvent.click(screen.getByRole("button", { name: /get risk flag/i }));
    expect(await screen.findByText("High Risk")).toBeInTheDocument();
    expect(mockedScore).toHaveBeenCalledTimes(1);
  });

  it("shows an error with a Retry button when the API fails", async () => {
    mockedScore.mockResolvedValue({
      status: 503,
      code: "MODEL_UNAVAILABLE",
      message: "down",
    } as never);
    render(<ScorePage />);
    fireEvent.click(screen.getByRole("button", { name: /load sample patient/i }));
    fireEvent.click(screen.getByRole("button", { name: /get risk flag/i }));
    expect(await screen.findByRole("alert")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /retry/i })).toBeInTheDocument();
  });
});
