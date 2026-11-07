import { FIELDS, SAMPLE_PATIENT, toFeatures, validateAll, validateField } from "./validation";

const spec = (name: string) => FIELDS.find((f) => f.name === name)!;

describe("validateField", () => {
  it("requires a value", () => expect(validateField(spec("age"), "")).toBe("Required"));
  it("rejects non-numbers", () => expect(validateField(spec("age"), "abc")).toBe("Enter a number"));
  it("rejects out-of-range values", () => {
    expect(validateField(spec("age"), "-5")).toMatch(/between 0 and 120/);
    expect(validateField(spec("thal"), "0")).toMatch(/between 1 and 3/);
  });
  it("rejects decimals for integer fields but allows them for oldpeak", () => {
    expect(validateField(spec("age"), "63.5")).toBe("Enter a whole number");
    expect(validateField(spec("oldpeak"), "2.3")).toBeNull();
  });
  it("accepts valid values", () => expect(validateField(spec("chol"), "233")).toBeNull());
});

describe("validateAll / toFeatures", () => {
  it("sample patient has no errors", () => expect(validateAll(SAMPLE_PATIENT)).toEqual({}));
  it("empty form flags every field", () =>
    expect(Object.keys(validateAll({})).length).toBe(FIELDS.length));
  it("converts strings to numbers", () => {
    const f = toFeatures(SAMPLE_PATIENT);
    expect(f.age).toBe(63);
    expect(f.oldpeak).toBeCloseTo(2.3);
  });
});
