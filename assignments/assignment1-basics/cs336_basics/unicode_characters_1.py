#>>> chr(0) #'\x00'
# text="hello world"
### text.encode("utf-8") # b'hello world'

import regex as re

PAT=r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

matches=re.findall(PAT,"some text that i'll pre-tokenize")
print(matches)

matches = re.finditer(PAT,"some text that i'll pre-tokenize")

for match in matches:
    print(match.group(),end=",")
print()


greatest_pair= max([("A", "B"), ("A", "C"), ("B", "ZZ"), ("BA", "A")]) 
print(f"greatest_pair is {greatest_pair}")

# generates bytes code
text="low"
text_byte_encoding="low".encode("utf-8")
print(f"text_byte_encoding is {text_byte_encoding},type is {type(text_byte_encoding)}")

tuple_text_byte_encoding= tuple(bytes([byte_text]) for byte_text in text.encode("utf-8"))

print(f"tuple_text_byte_encoding is {tuple_text_byte_encoding},type is {type(tuple_text_byte_encoding)}")


print("hello world".split())
print("hello world".split(" "))

###
"""
Differences:
split() (no arguments):
Splits on any whitespace (spaces, tabs, newlines)
Removes empty strings from the result
Handles multiple consecutive spaces/newlines cleanly
Example: "hello world" → ['hello', 'world'] (no empty strings)
split(" ") (with space argument):
Splits only on single space characters
Keeps empty strings in the result
Doesn't split on tabs or newlines
Example: "hello world" → ['hello', '', 'world'] (empty string between)
For your BPE training:
Use split() because:
It handles newlines in your corpus (see Test 5)
It removes empty strings
It splits on all whitespace, not just spaces
You get clean word lists: ['low', 'low', 'low', ...]
"""
###

test_tuple = (b'l', b'o', b'w',b'e',b'r')
# Try: list(zip(test_tuple, test_tuple[1:]))
print(list(zip(test_tuple, test_tuple[1:])))