import { act, fireEvent, render, screen } from "@testing-library/react";
import { vi } from "vitest";
import ScorePage from "./ScorePage";
import { scorePatient } from "../services/api";

vi.mock("../services/api", () => ({
  scorePatient: vi.fn(),
  toApiFailure: () => ({ status: 503, code: "MODEL_UNAVAILABLE", message: "down" }),
}));

const mockedScore = vi.mocked(scorePatient);

describe("ScorePage", () => {
  // Braces matter: Vitest runs a returned function (the mock itself) as cleanup.
  beforeEach(() => {
    mockedScore.mockReset();
  });

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
      risk_probability: 0.8, risk_flag: "High Risk", model_version: "m2.0.0", voting_strategy: "soft",
      threshold: 0.4, model_used: "full", missing_tests: [], notice: null, timestamp: "2026-01-01T00:00:00Z", disclaimer: "Not a diagnosis.",
    });
    render(<ScorePage />);
    fireEvent.click(screen.getByRole("button", { name: /load sample patient/i }));
    fireEvent.click(screen.getByRole("button", { name: /get risk flag/i }));
    expect(await screen.findByText("High Risk")).toBeInTheDocument();
    expect(mockedScore).toHaveBeenCalledTimes(1);
  });

  it("sends null when a scan is marked as not done", () => {
    render(<ScorePage />);
    fireEvent.click(screen.getByRole("button", { name: /load sample patient/i }));
    fireEvent.change(screen.getByLabelText(/major vessels/i), { target: { value: "not_done" } });
    fireEvent.click(screen.getByRole("button", { name: /get risk flag/i }));
    expect(mockedScore).toHaveBeenCalledWith(expect.objectContaining({ ca: null, thal: 1 }));
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

  describe("when a value changes after submitting", () => {
    const result = (flag: "High Risk" | "Low Risk") => ({
      risk_probability: 0.8, risk_flag: flag, model_version: "m2.0.0", voting_strategy: "soft",
      threshold: 0.4, model_used: "full" as const, missing_tests: [], notice: null,
      timestamp: "2026-01-01T00:00:00Z", disclaimer: "Not a diagnosis.",
    });
    const click = (name: RegExp) => fireEvent.click(screen.getByRole("button", { name }));
    const submitSample = () => {
      render(<ScorePage />);
      click(/load sample patient/i);
      click(/get risk flag/i);
    };
    const changed = /values changed/i;

    it.each([
      ["a number field", () => fireEvent.change(screen.getByLabelText(/^age/i), { target: { value: "50" } })],
      ["a select field", () => fireEvent.change(screen.getByLabelText(/chest pain/i), { target: { value: "0" } })],
      ["loading the sample patient", () => click(/load sample patient/i)],
    ])("removes the result: %s", async (_, edit) => {
      mockedScore.mockResolvedValue(result("High Risk"));
      submitSample();
      await screen.findByText("High Risk");
      expect(screen.queryByText(changed)).toBeNull();
      edit();
      expect(screen.queryByText("High Risk")).toBeNull();
      expect(screen.getByText(changed)).toBeInTheDocument();
      expect(mockedScore).toHaveBeenCalledTimes(1);
    });

    it("shows the new result, without the hint, after submitting again", async () => {
      mockedScore.mockResolvedValueOnce(result("High Risk")).mockResolvedValueOnce(result("Low Risk"));
      submitSample();
      await screen.findByText("High Risk");
      fireEvent.change(screen.getByLabelText(/^age/i), { target: { value: "50" } });
      click(/get risk flag/i);
      expect(await screen.findByText("Low Risk")).toBeInTheDocument();
      expect(screen.queryByText(changed)).toBeNull();
      expect(mockedScore).toHaveBeenLastCalledWith(expect.objectContaining({ age: 50 }));
    });

    it("ignores the answer to a request that was edited while running", async () => {
      let answer: (r: ReturnType<typeof result>) => void = () => undefined;
      mockedScore.mockReturnValue(new Promise((resolve) => { answer = resolve; }) as never);
      submitSample();
      fireEvent.change(screen.getByLabelText(/^age/i), { target: { value: "50" } });
      await act(async () => answer(result("High Risk")));
      expect(screen.queryByText("High Risk")).toBeNull();
      expect(screen.getByText(changed)).toBeInTheDocument();
    });

    it("removes an error too", async () => {
      mockedScore.mockRejectedValue(new Error("down"));
      submitSample();
      await screen.findByRole("alert");
      fireEvent.change(screen.getByLabelText(/^age/i), { target: { value: "50" } });
      expect(screen.queryByRole("alert")).toBeNull();
    });

    it("clears everything on reset, without the hint", async () => {
      mockedScore.mockResolvedValue(result("High Risk"));
      submitSample();
      await screen.findByText("High Risk");
      click(/reset form/i);
      expect(screen.queryByText("High Risk")).toBeNull();
      expect(screen.queryByText(changed)).toBeNull();
    });
  });

  it("explains a slow response", () => {
    vi.useFakeTimers();
    try {
      mockedScore.mockReturnValue(new Promise(() => undefined) as never);
      render(<ScorePage />);
      fireEvent.click(screen.getByRole("button", { name: /load sample patient/i }));
      fireEvent.click(screen.getByRole("button", { name: /get risk flag/i }));
      expect(screen.getByText(/analyzing/i)).toBeInTheDocument();
      act(() => { vi.advanceTimersByTime(4100); });
      expect(screen.getByText(/waking up/i)).toBeInTheDocument();
    } finally {
      vi.useRealTimers();
    }
  });
});
