====================
Example stats config
====================

.. code-block:: yaml

    init:
    # From Stats
    dataset: bom_ww3_oceania
    intake_catalog: /config/catalog/intake/oceanum.yml
    mask: self.dset.hs<=0
    slice_dict:
        sel:
        time: !!python/object/apply:slice ["1979-01-10", null]
    # From DerivedVar
    hs_threshold: 2.0
    var_hs: hs
    var_hs_sea: hs0
    var_hs_sw1: hs1
    var_hs_sw2: hs2
    var_dir_sea: th0
    var_dir_sw1: th1
    var_dir_sw2: th2
    var_lp_sw1: lp1
    var_uwnd: U10
    var_vwnd: V10

    calls:
    - method: time_stat
        kwargs:
        func: mean
        data_vars:
            - hs
        derived_vars:
            - wspd

    - method: time_probability
        kwargs:
        derived_vars:
            - douglas_sea
            - douglas_swell
        bins: [0,1,2,3,4,5,6,7,8,9]
        bin_name: scale

    - method: time_probability
        kwargs:
        derived_vars:
            - crossing_seas
        bins: [True] # crossing_seas is a bool dataarray

    - method: time_quantile
        kwargs:
        data_vars:
            - hs
        derived_vars:
            - wspd
        q: [0.5, 0.75, 0.9, 0.95, 0.99]

    - method: to_netcdf
        kwargs:
        outfile: /data/gridstat-bom-ww3-oceania.nc
        format: NETCDF4_CLASSIC
        zlib: True
        _FillValue: -32767
