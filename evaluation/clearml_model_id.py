# Description: This file contains the model ids for the trained models on ClearML.

unet_ids = {  # num_training_samples: model_id
    1: '1ab541353f2d4c14b8fc21ea97a09500',
    5: 'f9da83e367494f13911a73099b9713f6',
    10: '4d8666c9add3450e8ba02529876c3085',
    15: '0cdf1b21adc64bc19a0a1d48c53b4e4c',
    20: 'c4f92410b12e481ea476fc16254594d1',
    25: '57a1cd9dd8e64a128ca21b44ab4d304f',
    30: '74392de2ad0242b5be59f07e6d9c2c0d',
    35: 'fc641b2799ec468dbdc5cf0478468086',
    43: 'bf9286353ce649ef880774f62715c100'
}

raw_pseudo_lbl_unet_ids = {  # num_training_samples: model_id
    1: 'e04b2c2424aa46109b6ac23e1359c7f0',
    5: '22a24f8b42dd4558b35cf6c2dc5c2198',
    10: 'e448fd16272f443aab04f2b529329606',
    15: '85ee67c718e24dacb0cd26170f1beb50',
    20: 'da1dd843b0254eaf846050900994f480',
    25: 'c896158541e848e882ff25d22e859cdd',
    30: 'dfa2fd6448ef4b4ebc903e28576cb8c8',
    35: '1dc80448f1264941bc801cfd1bdfa610',
    43: '2bc2147a5bc744c694e3ab98277ba168'
}

sam_pseudo_lbl_unet_ids = {  # num_training_samples: model_id
    1: '288731644dbb4036bc31b11ccd5cac1d',
    5: '57e00999d27a4d7da3750a5451c8dfde',
    10: '1fa28cc268ef41529a44f3586aa33baf',
    15: '7a343db6847d4d5181d10b06ba07cf03',
    20: 'aa309964f0e44728bdfa16624a0dcf93',
    25: '457a707a49fd43748458241e58478636',
    30: 'e0e6bb59958d40bdaef9e55b7827da09',
    35: '664b27b3c5d34d7fbb96b15e1ee18064',
    43: 'daae93c8731f4914b0c88278076dc192'
}

sam_lraspp = {  # num_training_samples: model_id
    1: '22e0816472e348d18c286888214d446d',
    5: '72d126346002453dbcd89a67793bc64c',
    10: 'fb70b43957554137b9f623714977bf4e',
    15: 'afc75bb44b594570a68a35906f268b19',
    20: '489e10cb521c499fa2b19ae7e1b5ff0d',
    25: '93219b2459854f05bafb6450f943f78c',
    30: 'e1b0e28d498c41b7b5a02e7ef3b18e12',
    35: '153cfe9be80246fbb3d03cfacdb84bd5',
    43: '949d370f12aa4f4cb7247d27e6bfc7c6'
}

unet_mean_teacher_ids = {  # num_training_samples: model_id
    1: '8fe49ef3d53d455c90e28f6b34d0da2c',
    5: 'bae55716644640a5b0292007debf5cd2',
    10: '5e4736967f68411287d5d33fb5429e65',
    15: '308c229e394f48ab9cb4e577184d2a49',
    20: 'b0c2a667530a435ebcef51ad851be9ac',
    25: 'b131347b733847d789cc077d2cb9221e',
    30: 'b626efb7d04b4170ad639001e901b228',
    35: 'ac23eae5dce94b1ab22f9c8aa4208321',
    43: 'a7364b31977e42a2a15ac511cfed358f',
}

dental_models = {
    'unet_45_lbl': 'fff060f575994796936422b8c2819c5e',
    'unet_all_lbl': 'df0fe1e79c384b498e8236bcd89f0c2b',
    'unet_raw_pseudo_lbl': '04678a5f6cd64c4bb6fce92074022a97',
    'unet_sam_pseudo_lbl': '274591116b004e348cfe34ac9608ba9e',
    'mean_teacher': '3bdbce9d81b24aa796c389cd2e188313',
    'mean_teacher_sam_selection': '50993098131d4087b11447131d45e125',
    'sam_lraspp': '9f9cfea833c84ce998cddcd8414a9985'
}
