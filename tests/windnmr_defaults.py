"""
Reich default values for multiplet calculations. Consider merging with nspin.py
"""

ABdict = {'Jab': 12.0,
          'Vab': 15.0,
          'Vcentr': 150.0}

AB2dict = {'Jab': 12.0,
           'Vab': 15.0,
           'Vcentr': 150.0}

# for expediency, including the 2,6-dichlorophenol AB2 for test_nmrmath
dcp = {'Jab': 7.9,
        'Vab': 26.5,
        'Vcentr': 13.25}

ABXdict = {'Jab': 12.0,
           'Jax': 2.0,
           'Jbx': 8.0,
           'Vab': 15.0,
           'Vcentr': 7.5}

AMX3dict = {'Jab': -12.0,
            'Jax': 7.0,
            'Jbx': 7.0,
            'Vab': 14.0,
            'Vcentr': 150}

ABX3dict = {'Jab': -12.0,
            'Jax': 7.0,
            'Jbx': 7.0,
            'Vab': 14.0,
            'Vcentr': 150}

AAXXdict = {"Jaa": 15.0,
            "Jxx": -10.0,
            "Jax": 40.0,
            "Jax_prime": 6.0,
            'Vcentr': 150}

AABBdict = {"Vab": 40,
            "Jaa": 15.0,
            "Jbb": -10.0,
            "Jab": 40.0,
            "Jab_prime": 6.0,
            'Vcentr': 150}

ab_kwargs = {'model': 'AB', 'vars': ABdict,
             'widgets': ['Jab', 'Vab', 'Vcentr']}

ab2_kwargs = {'model': 'AB2', 'vars': AB2dict,
              'widgets': ['Jab', 'Vab', 'Vcentr']}

abx_kwargs = {'model': 'ABX', 'vars': ABXdict,
              'widgets': ['Jab', 'Jax', 'Jbx', 'Vab', 'Vcentr']}

abx3_kwargs = {'model': 'ABX3', 'vars': ABX3dict,
               'widgets': ['Jab', 'Jax', 'Jbx', 'Vab', 'Vcentr']}

aaxx_kwargs = {'model': 'AAXX', 'vars': AAXXdict,
               'widgets': ['Jaa', 'Jxx', 'Jax', 'Jax_prime', 'Vcentr']}

aabb_kwargs = {'model': 'AABB', 'vars': AABBdict,
               'widgets': ['Jaa', 'Jbb', 'Jab', 'Jab_prime', 'Vcentr']}

multiplet_bar_defaults = {'AB': ab_kwargs,
                          'AB2': ab2_kwargs,
                          'ABX': abx_kwargs,
                          'ABX3': abx3_kwargs,
                          'AAXX': aaxx_kwargs,
                          'AABB': aabb_kwargs}


if __name__ == '__main__':
    for bar in ['AB', 'AB2', 'ABX', 'ABX3', 'AAXX', 'AABB']:
        print(multiplet_bar_defaults[bar]['model'])
        print(multiplet_bar_defaults[bar]['vars'])
        print(multiplet_bar_defaults[bar]['widgets'])
