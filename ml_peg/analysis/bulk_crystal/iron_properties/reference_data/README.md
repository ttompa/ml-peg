# Iron Properties reference data

The curve files are copied from the public implementation accompanying Zhang et al.,
*Acta Materialia* 270 (2024) 119788:
`https://github.com/leiapple/Potential_benchmark_iron`.

- `bain.csv`: `pot_testing/REF_DATA/BainPath_DFT.csv`
- `eos.csv`: `pot_testing/REF_DATA/eos_dft.csv`
- `sfe_110.csv`: `pot_testing/REF_DATA/110_111.csv`
- `sfe_112.csv`: `pot_testing/REF_DATA/112_111.csv`
- `ts_100.csv`: `pot_testing/REF_DATA/ts_100_dft.csv`
- `ts_110.csv`: `pot_testing/REF_DATA/ts_110_dft.csv`

Headers were added and whitespace-only numeric formatting was normalized. The EOS
file retains its decimal-comma representation and is parsed accordingly. Analysis
subtracts the BCC Bain energy and the minimum EOS energy before comparison.

`reference_values.json` records scalar values, units, sources, and the vacancy
supercell protocol. The thermomechanical, vacancy, and surface references come from
Dragoni et al., *Physical Review Materials* 2 (2018) 013808.
