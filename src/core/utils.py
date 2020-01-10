
def camelcase_to_snake_format(input: str) -> str:
    output = ''
    prev_c = ''
    for c in input:
        if c.isupper():
            output += '_' + c.lower() if prev_c.isalpha() else c.lower()
        else:
            output += c
        prev_c = c
    return output
