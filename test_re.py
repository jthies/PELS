#/*******************************************************************************************/
#/* This file is part of the training material available at                                 */
#/* https://github.com/jthies/PELS                                                          */
#/* You may redistribute it and/or modify it under the terms of the BSD-style licence       */
#/* included in this software.                                                              */
#/*                                                                                         */
#/* Contact: Jonas Thies (j.thies@tudelft.nl)                                               */
#/*                                                                                         */
#/*******************************************************************************************/

import re

input = 'Laplace128x256'
input_list = re.match(r"Laplace(?P<nx>[-+]?\d+)x(?P<ny>[-+]?\d+)", input)
print(f"[{input_list['nx']}, {input_list['ny']}]")


