import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def convert_string_tuple_to_tuple(repr, cast_func=int):
    return tuple(cast_func(item) for item in repr[1:-1].split(',') if item != '')

if __name__ == '__main__':

    mdf_data = pd.read_csv('mdf_bands.csv', index_col=False)
    mdf_data.sort_values(by=['times_played', 'band_name'], inplace=True, ascending=[False,True])
    mdf_data.set_index('band_name')

    print(mdf_data)

    heads = mdf_data.sample(100)

    bands = heads['band_name']
    sets = heads['times_played']

    all_years = []
    for year in mdf_data['years_played']:
        all_years += list(convert_string_tuple_to_tuple(year))
    

    unique_years = list(set(all_years))
    unique_years.sort()

    year_occurence = []
    for unique_year in unique_years:
        year_occurence += [all_years.count(unique_year)]

    fig, (ax1,ax2) = plt.subplots(2,1,figsize=(20,20))

    ax1.bar(bands, sets)
    ax1.tick_params(axis='x', labelrotation=90)
    ax1.grid(True, which='both', axis='y')
    ax1.set_xlim(-.5, heads.index.size-.5)
    ax1.set_yticks(range(0,sets.max()+1))

    ax2.bar(unique_years, year_occurence)
    ax2.set_xticks(unique_years)
    ax2.grid(True, which='both', axis='y')
    ax2.set_xlim(unique_years[0]-.5, unique_years[-1]+.5)
    

    fig.tight_layout(pad=2)

    plt.show()