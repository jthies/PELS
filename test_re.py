import re

input = 'Laplace128x256'
input_list = re.match(r"Laplace(?P<nx>[-+]?\d+)x(?P<ny>[-+]?\d+)", input)
print(f"[{input_list['nx']}, {input_list['ny']}]")


