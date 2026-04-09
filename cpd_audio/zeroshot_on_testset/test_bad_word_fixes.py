from normalization import fix_bad_words, get_bad_word_fixes

def test_fix_bad_words_csv():
    passed = 0
    failed = 0
    fixes = get_bad_word_fixes()  

    def run_test(name, text, expected):
        nonlocal passed, failed
        result = fix_bad_words(text, fixes)
        if result == expected:
            print(f"  PASS: {name}")
            passed += 1
        else:
            print(f"  FAIL: {name}")
            print(f"        Input:    {repr(text)}")
            print(f"        Expected: {repr(expected)}")
            print(f"        Got:      {repr(result)}")
            failed += 1

    run_test("Single digit 0",               "the suspect had 0 priors",                    "the suspect had ZERO priors")
    run_test("Single digit 1",               "only 1 unit responded",                       "only ONE unit responded")
    run_test("Two digit number 10",          "10 minutes ago",                              "TEN minutes ago")
    run_test("Three digit 100",              "block of 100 Main Street",                    "block of ONE HUNDRED Main Street")
    run_test("Four digit 1000",              "approximately 1000 feet away",                "approximately ONE THOUSAND feet away")
    run_test("Time-like value 1200",         "occurred at 1200 hours",                      "occurred at TWELVE HUNDRED hours")
    run_test("Time-like value 1900",         "last seen at 1900 hours",                     "last seen at NINETEEN HUNDRED hours")
    run_test("10TH -> TENTH",               "he lives on the 10TH floor",                  "he lives on the TENTH floor")
    run_test("185TH -> ONE HUNDRED...",     "the 185TH district",                          "the ONE HUNDRED AND EIGHTY FIFTH district")
    run_test("0'S -> ZERO'S",              "a pair of 0'S on the plate",                   "a pair of ZERO'S on the plate")
    run_test("10'S -> TEN'S",              "running in the 10'S",                           "running in the TEN'S")
    run_test("13'S -> THIRTEEN'S",         "two 13'S in the sequence",                      "two THIRTEEN'S in the sequence")
    run_test("13'CAUSE -> THIRTEEN 'CAUSE","13'CAUSE that's the code",                      "THIRTEEN 'CAUSE that's the code")
    run_test("0KAY -> OKAY",               "0KAY we are en route",                         "OKAY we are en route")
    run_test("FOURTY -> FORTY",            "FOURTY suspects fled",                          "FORTY suspects fled")
    run_test("OK -> OKAY",                 "OK we are en route",                            "OKAY we are en route")
    run_test("O -> OH",                    "unit O nine responding",                        "unit OH nine responding")
    run_test("DOB -> D O B",               "requesting DOB on suspect",                     "requesting D O B on suspect")
    run_test("EMS -> E M S",               "EMS is on scene",                               "E M S is on scene")
    run_test("DONT -> DON'T",              "DONT approach the vehicle",                     "DON'T approach the vehicle")
    run_test("EDDY -> EDDIE",              "suspect named EDDY",                            "suspect named EDDIE")
    run_test("00 passes through",          "unit 00 responding",                            "unit 00 responding")
    run_test("002 passes through",         "code 002 issued",                               "code 002 issued")
    run_test("'KAY passes through",        "yes 'KAY understood",                           "yes 'KAY understood")
    run_test(
        "Mixed numbers and regular words",
        "10 units responding to 100 Main at 1200 hours",
        "TEN units responding to ONE HUNDRED Main at TWELVE HUNDRED hours"
    )
    run_test(
        "Hardcoded fixes in context",
        "FOURTY OK DONT approach suspect at 100 Main",
        "FORTY OKAY DON'T approach suspect at ONE HUNDRED Main"
    )
    run_test(
        "Case insensitivity on alpha-numeric",
        "0kay we copy",
        "OKAY we copy"
    )

    print(f"\nResults: {passed} passed, {failed} failed")


if __name__ == "__main__":
    test_fix_bad_words_csv()