ph_idx_to_type = [    'Aromatic',
    'HydrogenDonor',
    'HydrogenAcceptor',
    'PositiveIon',
    'NegativeIon',
    'Hydrophobic',
    'Halogen',
    ]

ph_type_to_idx = { val:idx for idx,val in enumerate(ph_idx_to_type) }