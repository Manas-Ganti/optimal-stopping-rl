"""Published reference values: Longstaff & Schwartz (2001), "Valuing American Options by
Simulation: A Simple Least-Squares Approach", Review of Financial Studies 14(1), Table 1.

K = 40, r = 0.06. `fd_american` is the paper's finite-difference American put value (the
Gate B reference); `european` is its closed-form European put (used to cross-check our
Black-Scholes); `lsm_paper` is the paper's own LSM estimate (50 exercise dates/yr,
100,000 paths), reported for context only.
"""

LS2001_STRIKE = 40.0
LS2001_RATE = 0.06
LS2001_DATES_PER_YEAR = 50

# (spot, vol, expiry, fd_american, european, lsm_paper)
LS2001_TABLE1 = [
    (36, 0.20, 1, 4.478, 3.844, 4.472),
    (36, 0.20, 2, 4.840, 3.763, 4.821),
    (36, 0.40, 1, 7.101, 6.711, 7.091),
    (36, 0.40, 2, 8.508, 7.700, 8.488),
    (38, 0.20, 1, 3.250, 2.852, 3.244),
    (38, 0.20, 2, 3.745, 2.991, 3.735),
    (38, 0.40, 1, 6.148, 5.834, 6.139),
    (38, 0.40, 2, 7.670, 6.979, 7.669),
    (40, 0.20, 1, 2.314, 2.066, 2.313),
    (40, 0.20, 2, 2.885, 2.356, 2.879),
    (40, 0.40, 1, 5.312, 5.060, 5.308),
    (40, 0.40, 2, 6.920, 6.326, 6.921),
    (42, 0.20, 1, 1.617, 1.465, 1.617),
    (42, 0.20, 2, 2.212, 1.841, 2.206),
    (42, 0.40, 1, 4.582, 4.379, 4.588),
    (42, 0.40, 2, 6.248, 5.736, 6.243),
    (44, 0.20, 1, 1.110, 1.017, 1.118),
    (44, 0.20, 2, 1.690, 1.429, 1.675),
    (44, 0.40, 1, 3.948, 3.783, 3.957),
    (44, 0.40, 2, 5.647, 5.202, 5.622),
]
