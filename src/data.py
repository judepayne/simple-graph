# TODO: data generator functionality


# g1_name is the example hierarchy used by the 'CRs in practice' document
g1_prime = {
    "nodes": [
        {"lei": "LEI00012047z12048", "name": "Vacfin", "type": "le", "level": 3},
        {"lei": "LEI00012037z12038", "name": "Investae", "type": "le", "level": 2},
        {"lei": "LEI00012027z12028", "name": "Feminance", "type": "le", "level": 2},
        {"lei": "LEI00012041z12042", "name": "MoonFund", "type": "le", "level": 2},
        {"lei": "LEI00012039z12040", "name": "FrostFinance", "type": "le", "level": 2},
        {"lei": "LEI00012045z12046", "name": "Calqulate", "type": "le", "level": 3},
        {"lei": "LEI00012043z12044", "name": "Finvac", "type": "le", "level": 3},
        {"lei": "LEI00012029z12030", "name": "Prosperous Ledger", "type": "le", "level": 2},
        {"lei": "LEI00012031z12032", "name": "Captal", "type": "le", "level": 2},
        {"lei": "LEI00012035z12036", "name": "Networth", "type": "le", "level": 2},
        {"lei": "LEI00012017z12018", "name": "Fiscaledge", "type": "le", "level": 0},
        {"lei": "LEI00012033z12034", "name": "The Finance Nerds", "type": "le", "level": 2},
        {"lei": "LEI00012021z12022", "name": "GuruFunds", "type": "le", "level": 1},
        {"lei": "LEI00012025z12026", "name": "MightyFunding", "type": "le", "level": 2},
        {"lei": "LEI00012023z12024", "name": "CynwitCaptial", "type": "le", "level": 1},
        {"lei": "LEI00012019z12020", "name": "FluidFinances", "type": "le", "level": 1},
    ],
    "edges": [
        {"src": "LEI00012023z12024", "dest": "LEI00012033z12034", "relationship-type": "legal"},
        {"src": "LEI00012017z12018", "dest": "LEI00012023z12024", "relationship-type": "legal"},
        {"src": "LEI00012019z12020", "dest": "LEI00012039z12040", "relationship-type": "legal"},
        {"src": "LEI00012021z12022", "dest": "LEI00012027z12028", "relationship-type": "legal"},
        {"src": "LEI00012021z12022", "dest": "LEI00012029z12030", "relationship-type": "legal"},
        {"src": "LEI00012023z12024", "dest": "LEI00012031z12032", "relationship-type": "legal"},
        {"src": "LEI00012025z12026", "dest": "LEI00012045z12046", "relationship-type": "legal"},
        {"src": "LEI00012023z12024", "dest": "LEI00012035z12036", "relationship-type": "legal"},
        {"src": "LEI00012025z12026", "dest": "LEI00012047z12048", "relationship-type": "legal"},
        {"src": "LEI00012019z12020", "dest": "LEI00012041z12042", "relationship-type": "legal"},
        {"src": "LEI00012021z12022", "dest": "LEI00012025z12026", "relationship-type": "legal"},
        {"src": "LEI00012025z12026", "dest": "LEI00012043z12044", "relationship-type": "legal"},
        {"src": "LEI00012017z12018", "dest": "LEI00012021z12022", "relationship-type": "legal"},
        {"src": "LEI00012019z12020", "dest": "LEI00012037z12038", "relationship-type": "legal"},
        {"src": "LEI00012017z12018", "dest": "LEI00012019z12020", "relationship-type": "legal"},
    ]
}

