import { FIELDS, NOT_DONE, SAMPLE_PATIENT, toFeatures, validateAll, validateField } from "./validation";

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

describe("clinical option labels", () => {
  const label = (name: string, value: number) =>
    spec(name).options!.find((o) => o.value === value)!.label;
  it("match the verified UCI recoding", () => {
    expect(label("cp", 0)).toBe("Asymptomatic");
    expect(label("cp", 3)).toBe("Typical angina");
    expect(label("restecg", 1)).toBe("Normal");
    expect(label("slope", 2)).toBe("Upsloping");
    expect(label("thal", 1)).toBe("Fixed defect");
    expect(label("thal", 2)).toBe("Normal");
    expect(label("thal", 3)).toBe("Reversible defect");
  });
});

describe("tests that were not done", () => {
  it("are sent as null for ca and thal", () => {
    const f = toFeatures({ ...SAMPLE_PATIENT, ca: NOT_DONE, thal: NOT_DONE });
    expect(f.ca).toBeNull();
    expect(f.thal).toBeNull();
    expect(validateAll({ ...SAMPLE_PATIENT, ca: NOT_DONE, thal: NOT_DONE })).toEqual({});
  });
  it("must be chosen explicitly and are refused for other fields", () => {
    expect(validateField(spec("ca"), "")).toBe("Required");
    expect(validateField(spec("chol"), NOT_DONE)).toBe("Enter a number");
  });
});
