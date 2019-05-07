import src.util as util
import constants as c
import pandas as pd
from .expansion import unemployment, hpi, cpi_index, sp_500, federal_fund_rate, inflation, corperate_govt_spread, Fama_french, state_gdp


def generate(df, generate_unemployment=True, generate_hpi=True, generate_cpi=True,
             generate_sp500=True, generate_ffr=True, generate_inf=True, generate_spread=True,
             generate_fama_french=True, generate_state_gdp=True):
    years = list(map(str, range(c.START_DATE.year, c.END_DATE.year + 1)))
    states = set(df['addr_state'])

    if generate_unemployment:
        unemployment.generate(states)

        # TODO: consider getting data for PR from a different source
    if generate_hpi:
        hpi.generate(states)

    if generate_sp500:
        sp_500.generate(years)

        # TODO: figure out how to implement
    if generate_cpi:
        cpi_index.generate(years)

    if generate_ffr:
        federal_fund_rate.generate(years)

    if generate_inf:
        inflation.generate(years)

    if generate_spread:
        corperate_govt_spread.generate(years)

    if generate_fama_french:
        Fama_french.generate_3_factors(years)

    if generate_state_gdp:
        state_gdp.generate(states)
