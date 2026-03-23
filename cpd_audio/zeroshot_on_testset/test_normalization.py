from zeroshot_on_testset import extensive_normalization

test_cases = [
    {
        "name": "Two Twenty (Space)",
        "input": "The bus leaves at two twenty pm.",
        "expected": "the bus leaves at 2 2 0 pm"
    },
    {
        "name": "String of numbers", 
        "input": "TWENTY ONE ZERO SIX SEVEN THREE THREE ZERO SIX SEVEN THREE THREE", 
        "expected": "2 1 0 6 7 3 3 0 6 7 3 3"
    }, 
    {
        "name": "Two-Twenty (Hyphen)",
        "input": "It costs two-twenty dollars.",
        "expected": "it costs 2 2 0 dollars"
    },
    {
        "name": "Complex Compound",
        "input": "Two hundred and twenty-two.",
        "expected": "2 2 2"
    },
    {
        "name": "Mixed Digits",
        "input": "Flight 747 is ready.",
        "expected": "flight 7 4 7 is ready"
    }
]

print(f"{'test name':<20} | {'status':<10}")
print("-" * 35)

for case in test_cases:
    result = extensive_normalization(case["input"], debug=True)
    status = "pass" if result == case["expected"] else "fail"
    print(f"{case['name']:<20} | {status}")
    print(f"   input:    {case['input']}")
    print(f"   expected: {case['expected']}")
    print(f"   got:      {result}\n")
