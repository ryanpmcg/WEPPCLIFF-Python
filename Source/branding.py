"""WEPPCLIFF branding and license agreement — port of WEPPCLIFF.R lines 19-80."""

from __future__ import annotations

import sys

from Source.config import LR

LOGO = """
________________________________________________________________________________
________________________________________________________________________________

 ___      ___ _______ _______  _______  _______ ___     _______ _______ _______
 | |  __  | | |  ___/ |  __  | |  __  | |  ___/ | |     |__ __| |  ___/ |  ___/
 | | /  \\ | | | |___  | |__| | | |__| | | |     | |       | |   | |___  | |___
 | |/ /\\ \\| | |  __/  |  ____| |  ____| | |     | |       | |   |  __/  |  __/
 \\   /  \\   / | |___  | |      | |      | |___  | |___   _| |_  | |     | |
  \\_/    \\_/  |_____| |_|      |_|      |_____| |_____| |_____| |_|     |_|

________________________________________________________________________________
________________________________________________________________________________
"""

LICENSE = """
  WEPP Climate File Formatter (WEPPCLIFF) Version 1.7
  Copyright (c) 2023 Ryan P. McGehee

  This program is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation (GNU GPL v3.0).

  This program is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.

  You should have received a copy of the GNU General Public License
  along with this program.  If not, see <http://www.gnu.org/licenses/>.

  Please provide an appropriate citation in any published work as follows:

  McGehee, R.P., D.C. Flanagan, and P. Srivastava. 2020. WEPPCLIFF:
      A command-line tool to process climate inputs for soil loss models.
      Journal of Open Source Software. https://doi.org/10.21105/joss.02029
      Repository available at: https://github.com/ryanpmcg/WEPPCLIFF

  Corresponding Author: R.P. McGehee
  E-mail: ryanpmcgehee@gmail.com"""


def print_weppcliff_logo() -> None:
    """Print the WEPPCLIFF logo to screen."""
    print(LR * 30, end="")
    print(LOGO, end="")


def print_license_agreement() -> None:
    """Print the license agreement to screen."""
    print(LR, end="")
    print(LICENSE, end="")
    print(LR, end="")


def prompt_license_agreement() -> None:
    """Prompt the user to accept the license agreement."""
    print_license_agreement()
    print(LR, "Do you accept these conditions? (Y/N)", LR, end="", sep="")
    response = sys.stdin.readline().strip()
    if response.upper() != "Y":
        raise SystemExit(
            "You must acknowledge your agreement to the license conditions "
            "with 'Y or y' to use this software."
        )