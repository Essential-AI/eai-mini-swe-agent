import re
from math_verify import parse, verify

def latex_extract(result: str) -> str:
    """Extract the answer from the result."""
    if "\\boxed{" not in result:
        return result
    match = re.search(r"\\boxed{(.*)}", result)
    if match:
        return match.group(1)
    return None

def gsm_extract(result: str) -> str:
    """Look for #### at the end of the result."""
    if "####" not in result:
        return result
    match = re.search(r"####(.*)", result)
    if match:
        return match.group(1).strip()
    return None

def rm_latex_text(text: str) -> str:
    if text is None:
        return None
    """Remove `\text{...}` from the text."""
    rmed = re.sub(r"\\text{(.*)}", r"\1", text)
    rmed = re.sub(r"\\textbf{(.*)}", r"\1", rmed)
    return rmed

def replace_dfrac_to_frac(text: str) -> str:
    if text is None:
        return None
    """Replace `\\dfrac` with `\\frac`."""
    return re.sub(r"\\dfrac", r"\\frac", text)

def rm_currency_symbols(text: str) -> str:
    if text is None:
        return None
    """Remove currency symbols from the text."""
    return re.sub(r"\\$", "", text)

def normalize_uncurly_fractions(text: str) -> str:
    if text is None:
        return None
    """Normalize uncurly fractions."""
    return re.sub(r"\\frac(.)(.)", r"\\frac{\1}{\2}", text)

def math_match(result: str | None, answer: str) -> bool:
    """Use math-verify to check if the result matches the answer."""
    if result is None:
        return False

    rm_spaces = lambda text: text.replace(" ", "") if text is not None else None
    result, answer = rm_spaces(result), rm_spaces(answer)
    # extract `\text{...}` from the text
    result, answer = rm_latex_text(result), rm_latex_text(answer)
    # change \dfrac to \frac
    result, answer = replace_dfrac_to_frac(result), replace_dfrac_to_frac(answer)
    # normalize uncurly fractions
    result, answer = normalize_uncurly_fractions(result), normalize_uncurly_fractions(answer)
    # remove currency symbols
    result, answer = rm_currency_symbols(result), rm_currency_symbols(answer)

    if result == answer:
        return True

    gold = parse(result)
    answer = parse(answer)
    rslt = verify(gold, answer)
    # print(f"gold: {gold}, answer: {answer}, rslt: {rslt}")
    return rslt