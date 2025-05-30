# Copyright 2025 The Water Institute of the Gulf
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import tempfile

import numpy as np

from AdcircSubgrid.lookup_table import LookupTable


def test_lookup_table_ccap() -> None:
    """
    Test the ccap lookup table
    """
    lut = LookupTable("ccap")
    table = lut.lookup_table()

    assert len(table) == 256
    assert np.isnan(table[0])
    assert np.isnan(table[1])
    assert table[2] == 0.120
    assert table[15] == 0.070
    assert table[23] == 0.030
    assert np.isnan(table[24])
    assert np.isnan(table[50])
    assert np.isnan(table[255])


def test_lookup_table_file() -> None:
    """
    Test the lookup table read from a file
    """
    table_file = [
        [1, 0.1],
        [2, 0.2],
        [3, 0.3],
        [4, 0.4],
        [5, 0.5],
        [6, 0.6],
    ]

    with tempfile.NamedTemporaryFile(mode="w", delete=True) as f:
        np.savetxt(f, table_file, fmt="%d, %f")
        f.flush()

        lut = LookupTable(f.name)
        table = lut.lookup_table()

        assert len(table) == 256
        assert np.isnan(table[0])
        assert table[1] == 0.1
        assert table[2] == 0.2
        assert table[3] == 0.3
        assert table[4] == 0.4
        assert table[5] == 0.5
        assert table[6] == 0.6
        assert np.isnan(table[7])
        assert np.isnan(table[255])
